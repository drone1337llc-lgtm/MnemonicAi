"""Pluggable reasoning backends used during consolidation.

The memory core only ever asks the LLM to do two things:
  * `summarize(texts)` -> a short gist string (systems-consolidation / schema)
  * `extract_facts(text)` -> a list of atomic semantic facts

`HeuristicLLM` implements both with rules so the whole system runs offline.
`OpenAICompatibleLLM` implements both by prompting any OpenAI-style
`/chat/completions` endpoint (LM Studio, OpenAI, Ollama, ...), and falls back
to the heuristics if the model output cannot be parsed.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Dict, List, Sequence

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_LINKING = (" is ", " are ", " was ", " were ", " has ", " have ", " had ",
            " will ", " can ", " uses ", " lives ", " likes ", " prefers ",
            " needs ", " wants ", " means ", " equals ", " costs ")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


class HeuristicLLM:
    """Offline, rule-based fallback. No network, no dependencies."""

    name = "heuristic"

    def chat(self, messages: Sequence[Dict[str, str]]) -> str:
        users = [m["content"] for m in messages if m.get("role") == "user"]
        return users[-1][:280] if users else ""

    def extract_facts(self, text: str) -> List[str]:
        facts: List[str] = []
        for sent in _sentences(text):
            low = " " + sent.lower() + " "
            has_link = any(k in low for k in _LINKING)
            has_num = any(ch.isdigit() for ch in sent)
            has_proper = bool(re.search(r"\b[A-Z][a-z]+", sent[1:]))  # a capitalized word mid-sentence
            if has_link or has_num or has_proper:
                facts.append(sent.rstrip(".") + ".")
        # de-duplicate preserving order
        seen = set()
        out = []
        for f in facts:
            key = f.lower()
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out

    def summarize(self, texts: Sequence[str]) -> str:
        # extractive: pick the most information-dense sentence across inputs
        sents: List[str] = []
        for t in texts:
            sents.extend(_sentences(t))
        if not sents:
            return ""
        sents.sort(key=lambda s: len(set(_tokenize(s))), reverse=True)
        return sents[0].rstrip(".") + "."


def _tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class OpenAICompatibleLLM:
    """Reasoning backend over an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = None,
                 timeout: float = 60.0, temperature: float = 0.2) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.name = "openai-compatible:" + model
        self._fallback = HeuristicLLM()

    def chat(self, messages: Sequence[Dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/chat/completions", data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def extract_facts(self, text: str) -> List[str]:
        prompt = (
            "Extract the atomic, standalone facts stated in the text below. "
            "Return ONLY a JSON array of short strings, no commentary.\n\nTEXT:\n" + text
        )
        try:
            out = self.chat([
                {"role": "system", "content": "You extract atomic facts as JSON."},
                {"role": "user", "content": prompt},
            ])
            facts = _parse_json_list(out)
            if facts:
                return facts
        except Exception:
            pass
        return self._fallback.extract_facts(text)

    def summarize(self, texts: Sequence[str]) -> str:
        joined = "\n- ".join(texts)
        prompt = ("Summarize the following related memories into ONE concise "
                  "sentence capturing their shared gist:\n- " + joined)
        try:
            out = self.chat([
                {"role": "system", "content": "You write concise one-sentence summaries."},
                {"role": "user", "content": prompt},
            ]).strip()
            if out:
                return out.split("\n")[0].strip()
        except Exception:
            pass
        return self._fallback.summarize(texts)


def _parse_json_list(text: str) -> List[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        return []

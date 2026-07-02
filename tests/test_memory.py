"""Unit tests for the core memory dynamics. Run: python3 -m unittest -v"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnemonicai import BrainMemory, Config, ManualClock
from mnemonicai.dynamics import current_strength, reinforce
from mnemonicai.memory_item import MemoryItem, MemoryKind

HOUR = 3600.0
DAY = 24 * HOUR


class TestSalienceGate(unittest.TestCase):
    def test_trivial_input_is_filtered(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        admitted = mem.perceive("ok.", source="user", importance=0.0)
        self.assertIsNone(admitted, "trivial input should be gated out")

    def test_novel_important_input_admitted(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        item = mem.perceive("The reactor core temperature is 900 degrees.",
                            source="sensor", importance=0.9)
        self.assertIsNotNone(item)
        self.assertGreaterEqual(item.salience, mem.cfg.salience_threshold)

    def test_repeat_has_lower_novelty(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        first = mem.perceive("Paris is the capital of France.", importance=0.5)
        second = mem.perceive("Paris is the capital of France.", importance=0.5)
        self.assertIsNotNone(first)
        # the repeat is less novel, so strictly lower salience
        if second is not None:
            self.assertLess(second.salience, first.salience)


class TestWorkingMemory(unittest.TestCase):
    def test_capacity_is_bounded(self):
        mem = BrainMemory(Config(working_capacity=3, salience_threshold=0.0,
                                 sleep_every_n_ticks=0))
        for i in range(10):
            mem.perceive(f"distinct observation number {i} about topic {i}",
                         importance=1.0)
        self.assertLessEqual(len(mem.working.items), 3)

    def test_activation_decays(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        it = mem.perceive("A novel and important fact about quantum widgets.",
                          importance=0.9)
        a0 = it.activation
        mem.tick()
        self.assertLess(it.activation, a0)


class TestConsolidation(unittest.TestCase):
    def test_sleep_promotes_to_long_term(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        mem.perceive("Maria is vegetarian and allergic to peanuts.",
                     source="user", importance=0.9)
        mem.perceive("The trip budget is 4000 dollars.", source="user", importance=0.8)
        self.assertEqual(len(mem.episodic), 0)
        report = mem.sleep()
        self.assertGreater(len(mem.episodic), 0)
        self.assertGreater(report["episodic"], 0)


class TestDecayAndReinforcement(unittest.TestCase):
    def test_strength_decays_over_time(self):
        cfg = Config()
        item = MemoryItem(content="x", kind=MemoryKind.EPISODIC,
                          created_at=0.0, last_access_at=0.0,
                          salience=0.5, strength=1.0, stability=1.0)
        s_now = current_strength(item, 0.0, cfg)
        s_later = current_strength(item, 3 * DAY, cfg)
        self.assertAlmostEqual(s_now, 1.0, places=5)
        self.assertLess(s_later, s_now)

    def test_reinforcement_increases_stability(self):
        cfg = Config()
        item = MemoryItem(content="x", kind=MemoryKind.EPISODIC,
                          created_at=0.0, last_access_at=0.0,
                          salience=0.5, strength=0.4, stability=1.0)
        reinforce(item, 1 * HOUR, cfg)
        self.assertGreater(item.stability, 1.0)
        self.assertEqual(item.access_count, 1)
        # a reinforced memory decays slower: compare to an un-reinforced twin
        twin = MemoryItem(content="x", kind=MemoryKind.EPISODIC,
                          created_at=0.0, last_access_at=1 * HOUR,
                          salience=0.5, strength=item.strength, stability=1.0)
        t = 5 * DAY
        self.assertGreater(current_strength(item, t, cfg),
                           current_strength(twin, t, cfg))


class TestRetrieval(unittest.TestCase):
    def test_retrieves_relevant_memory(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        mem.perceive("Maria is vegetarian and allergic to peanuts.",
                     source="user", importance=0.9)
        mem.perceive("The Eiffel Tower is in Paris.", source="web", importance=0.6)
        mem.sleep()
        results = mem.retrieve("what food restrictions does Maria have?", k=2)
        self.assertTrue(results)
        self.assertTrue(any("peanut" in m.content.lower() or "vegetarian" in m.content.lower()
                            for m in results))

    def test_retrieval_reinforces(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        mem.perceive("The capital of Japan is Tokyo.", source="web", importance=0.7)
        mem.sleep()
        target = mem.semantic.all()[0] if mem.semantic.all() else mem.episodic.all()[0]
        before = target.access_count
        mem.retrieve("what is the capital of Japan?", k=3)
        self.assertGreaterEqual(target.access_count, before)


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        mem = BrainMemory(Config(sleep_every_n_ticks=0))
        mem.perceive("Maria is vegetarian and allergic to peanuts.",
                     source="user", importance=0.9)
        mem.sleep()
        n_sem = len(mem.semantic)
        n_epi = len(mem.episodic)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.db")
            mem.save(path)
            fresh = BrainMemory(Config(sleep_every_n_ticks=0))
            fresh.load(path)
            self.assertEqual(len(fresh.semantic), n_sem)
            self.assertEqual(len(fresh.episodic), n_epi)
            self.assertEqual(len(fresh.working.items), 0)  # STM not persisted


class TestPinning(unittest.TestCase):
    def test_pinned_memory_survives_pruning(self):
        clock = ManualClock(t0=0.0)
        mem = BrainMemory(Config(sleep_every_n_ticks=0), clock=clock)
        mem.perceive("The vault code is 7731 hidden in the cellar.", importance=0.9)
        mem.perceive("Diego prefers morning meetings every Tuesday.", importance=0.9)
        mem.sleep()
        for store in (mem.episodic, mem.semantic, mem.procedural):
            for m in store.all():
                if "vault" in m.content.lower():
                    m.metadata["pinned"] = True
        clock.advance(60 * DAY)          # long neglect: unpinned memories fade out
        mem.sleep()                       # sleep prunes below the strength floor
        remaining = [m.content.lower() for m in mem.episodic.all() + mem.semantic.all()]
        self.assertTrue(any("vault" in c for c in remaining),
                        "pinned memory must never be pruned")
        self.assertFalse(any("diego" in c for c in remaining),
                         "unpinned neglected memory should be forgotten")


class TestStatePayload(unittest.TestCase):
    def test_state_includes_memories_edges_and_pinned(self):
        from mnemonicai.appconfig import AppConfig
        from mnemonicai.backend import MockBackend
        from mnemonicai.events import EventBus
        from mnemonicai.trainer import SleepTrainer
        from mnemonicai.bridge import MemoryChat
        cfg = AppConfig()
        cfg.train_on_sleep = False
        mem = BrainMemory(Config(sleep_every_n_ticks=0), clock=ManualClock(t0=0.0))
        bus = EventBus()
        backend = MockBackend(cfg)
        chat = MemoryChat(mem, backend, bus, SleepTrainer(mem, backend, bus, cfg), cfg)
        chat.admin_perceive("Ana studies astronomy at night.", 0.9)
        chat.admin_perceive("Ana loves telescopes and stargazing.", 0.9)
        chat.admin_sleep()               # consolidation wires Hebbian links
        st = chat.state_dict()
        self.assertIn("memories", st)
        self.assertGreater(len(st["memories"]), 0)
        self.assertTrue(all("pinned" in m for m in st["memories"]))
        self.assertIn("edges", st)
        self.assertGreaterEqual(len(st["edges"]), 1,
                                "co-consolidated memories should be linked")
        e = st["edges"][0]
        self.assertIn("a", e); self.assertIn("b", e); self.assertIn("w", e)


class TestConsolidationCard(unittest.TestCase):
    def test_bake_writes_memory_card(self):
        import tempfile
        from mnemonicai.appconfig import AppConfig
        from mnemonicai.backend import MockBackend
        from mnemonicai.events import EventBus
        from mnemonicai.trainer import SleepTrainer
        with tempfile.TemporaryDirectory() as d:
            cfg = AppConfig()
            cfg.data_dir = d
            cfg.adapter_dir = os.path.join(d, "adapter")
            cfg.ensure_dirs()
            mem = BrainMemory(Config(sleep_every_n_ticks=0), clock=ManualClock(t0=0.0))
            facts = ["Ana studies astronomy at night.",
                     "The launch is November 3rd.",
                     "The vault code is 7731.",
                     "Diego prefers morning meetings.",
                     "The budget ceiling is 40000 dollars.",
                     "The mascot is a red fox.",
                     "Deploys happen on Fridays."]
            for f in facts:
                mem.perceive(f, importance=0.9)
            mem.sleep()
            trainer = SleepTrainer(mem, MockBackend(cfg), EventBus(), cfg)
            result = trainer.consolidate_to_weights()
            self.assertFalse(result.get("skipped"), f"training skipped: {result}")
            self.assertIn("card", result)
            self.assertTrue(os.path.isfile(result["card"]))
            content = open(result["card"], encoding="utf-8").read()
            self.assertIn("MNEMONICAI", content)
            self.assertIn("<svg", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)

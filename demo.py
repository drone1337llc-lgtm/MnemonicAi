"""MnemonicAi demo: watch an agent form, consolidate, reinforce, and forget.

Runs fully offline (HashingEmbedder + HeuristicLLM). Uses a ManualClock so the
forgetting curve is visible over simulated hours and days.

    python3 demo.py
"""
from __future__ import annotations

from typing import Optional

from mnemonicai import BrainMemory, Config, ManualClock
from mnemonicai.memory_item import MemoryItem

HOUR = 3600.0
DAY = 24 * HOUR


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_working(mem: BrainMemory) -> None:
    items = sorted(mem.working.snapshot(), key=lambda i: i.activation, reverse=True)
    if not items:
        print("  (working memory empty)")
    for it in items:
        print(f"  [activation {it.activation:0.2f} | salience {it.salience:0.2f}] {it.short()}")


def show_store(mem: BrainMemory, store, label: str) -> None:
    items = store.all()
    print(f"  {label}: {len(items)} item(s)")
    for it in sorted(items, key=lambda i: mem.recall_strength(i), reverse=True):
        tag = "  <gist>" if it.summary_of else ""
        print(f"    - strength {mem.recall_strength(it):0.2f} | recalls {it.access_count}"
              f" | {it.short(52)}{tag}")


def find(store, needle: str) -> Optional[MemoryItem]:
    for it in store.all():
        if needle in it.content.lower():
            return it
    return None


def main(mem=None, clock=None) -> None:
    if mem is None:
        clock = ManualClock(t0=1_000_000.0)
        mem = BrainMemory(Config(sleep_every_n_ticks=0), clock=clock)  # manual sleep
    elif clock is None:
        clock = mem.clock  # must be a ManualClock so time can be advanced
    mem.set_goal("help the user plan a healthy trip to Japan")

    hr("1. PERCEPTION + SALIENCE GATE  (what is worth encoding?)")
    stream = [
        ("The user, Maria, is planning a two-week trip to Japan in October.", "user", 0.8),
        ("Maria is vegetarian and severely allergic to peanuts.", "user", 0.9),
        ("um, ok.", "user", 0.0),                          # filler -> filtered
        ("The user, Maria, is planning a two-week trip to Japan in October.", "user", 0.8),  # repeat -> low novelty
        ("Maria's budget for the trip is about 4000 dollars.", "user", 0.7),
        ("ok sounds good.", "user", 0.0),                  # filler -> filtered
        ("Kyoto has many vegetarian temple restaurants serving shojin ryori.", "tool:web", 0.6),
        ("To get a Japan Rail Pass: buy the voucher online, then exchange it at the airport station.",
         "action:booking", 0.7),                           # a skill / procedure
    ]
    for text, source, importance in stream:
        meta = {"skill": True} if source.startswith("action") else None
        item = mem.perceive(text, source=source, importance=importance, metadata=meta)
        verdict = "ADMITTED" if item else "filtered (forgotten)"
        print(f"  {verdict:20s} <- \"{text[:56]}\"")

    hr("2. WORKING MEMORY  (short-term, capacity-bounded, decaying)")
    show_working(mem)

    hr("3. MAINTENANCE REHEARSAL vs DECAY  (over ~2 hours)")
    print("  Maria keeps mentioning the allergy, so the agent rehearses it each step.")
    for step in range(5):
        mem.tick(dt=0.5 * HOUR)              # 30-min steps: activation decays
        allergy = next((i for i in mem.working.snapshot() if "allergic" in i.content), None)
        if allergy:
            mem.rehearse(allergy.id)         # actively kept alive
    print("\n  working memory after decay + selective rehearsal:")
    show_working(mem)
    print("  -> the rehearsed allergy trace stays active; the rest have faded toward staging.")

    hr("4. SLEEP  (consolidate short-term -> long-term)")
    report = mem.sleep()
    print(f"  consolidation report: {report}")
    show_store(mem, mem.episodic, "EPISODIC   (events that happened)")
    show_store(mem, mem.semantic, "SEMANTIC   (distilled facts + gists)")
    show_store(mem, mem.procedural, "PROCEDURAL (skills / how-to)")

    hr("5. SPACED REPETITION  (recall strengthens + stabilizes a memory)")
    # Use the memory's own text as the cue so retrieval deterministically returns
    # it, and track the *actual* reinforced object so stability growth is visible.
    seed = "Maria is vegetarian and severely allergic to peanuts."
    control_fact = find(mem.episodic, "kyoto") or find(mem.semantic, "kyoto")
    allergy_fact = None
    print("  The agent is asked about Maria's diet several times, spaced over ~1.5 days.")
    print(f"  {'recall #':>9} | {'strength':>8} | {'stability (durability)':>22}")
    for i in range(4):
        results = mem.retrieve(seed, k=1)
        allergy_fact = results[0]
        print(f"  {i + 1:>9} | {allergy_fact.strength:>8.2f} | {allergy_fact.stability:>22.2f}")
        clock.advance(8 * HOUR)              # spaced, not massed
    print(f"\n  allergy trace: recalled {allergy_fact.access_count}x  |  "
          f"control (kyoto) never recalled: {control_fact.access_count}x")
    print("  -> strength saturates near 1.0, but stability keeps compounding, which is")
    print("     what flattens the future forgetting curve.")

    hr("6. THE FORGETTING CURVE  (one week passes, no further recall)")
    print("  A reinforced memory outlives an un-rehearsed one (the spacing effect).")
    print(f"  {'day':>4} | {'allergy (recalled 4x)':>28} | {'kyoto (never recalled)':>22}")
    for day in range(0, 8):
        clock.advance(DAY if day > 0 else 0)
        a = mem.recall_strength(allergy_fact)
        c = mem.recall_strength(control_fact)
        bar_a = "#" * int(round(a * 20))
        bar_c = "#" * int(round(c * 20))
        print(f"  {day:>4} | {a:>6.2f} {bar_a:<20} | {c:>6.2f} {bar_c}")

    hr("7. PRUNING  (weak, unused traces are forgotten)")
    clock.advance(10 * DAY)                   # a week and a half more with no recall
    before = mem.stats()
    report = mem.sleep()                     # sleep also prunes below the strength floor
    after = mem.stats()
    print(f"  pruned {report['pruned']} faded memories")
    print(f"  before: episodic={before['episodic']} semantic={before['semantic']} "
          f"procedural={before['procedural']}")
    print(f"  after : episodic={after['episodic']} semantic={after['semantic']} "
          f"procedural={after['procedural']}")
    print(f"  allergy trace survived: {mem.episodic.get(allergy_fact.id) is not None} "
          f"(recalled {allergy_fact.access_count}x, so protected from pruning)")

    hr("8. PERSISTENCE  (long-term survives, short-term does not)")
    mem.save("agent_memory.db")
    fresh = BrainMemory(clock=ManualClock(t0=clock()))
    fresh.load("agent_memory.db")
    print(f"  reloaded: {fresh.stats()}")
    print("  (working memory is empty after reload, exactly like the brain)")

    print("\nThis same model is what the web simulation animates and the "
          "architecture blueprint specifies.\n")


if __name__ == "__main__":
    main()

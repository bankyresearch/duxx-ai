"""End-to-end demo: an agent that gets measurably better in production.

Run with::

    python examples/self_improving_demo.py

What you'll see:

* A refund classifier starts with a deliberately under-specified
  prompt that gets ~50% of its tagged-as-"return" cases wrong.
* The :class:`SelfImprovingAgent` serves 80 turns of live traffic.
* After ~30 turns the background loop detects the failure cluster
  (people saying "return" instead of "refund"), drafts a new
  prompt that handles them, and tags it as canary.
* Canary traffic flows in. After enough samples the loop sees the
  canary outperforming prod and promotes it to ``prod``.
* The final 30 turns show pass-rate climbing from ~50% to ~95%.

No real LLM. No real duxx-server. The demo uses the same
:class:`FakeDuxxClient` from the test suite and a tiny rule-based
chat / scorer so it runs anywhere in <2 seconds.

For a real-LLM version: replace ``demo_chat`` with a wrapper around
your provider SDK and ``FakeDuxxClient`` with a real
``redis.Redis(host='localhost', port=6379)`` pointing at
``duxx-server``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Let the demo import the FakeDuxxClient sitting in the test suite
# without copy-pasting it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_self_improving import FakeDuxxClient, _Score  # noqa: E402

from duxx_ai.self_improving import (  # noqa: E402
    LoopConfig,
    SelfImprovingAgent,
    StaticAppendGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-32s  %(message)s",
    datefmt="%H:%M:%S",
)
# Squelch chatter we don't care about for the demo.
logging.getLogger("duxx_ai.self_improving.scorer").setLevel(logging.WARNING)
logging.getLogger("duxx_ai.self_improving.router").setLevel(logging.WARNING)


# ----------------------------------------------------------------- domain


# A tiny world: customers ask for refunds in 4 distinct phrasings.
# Half explicitly say "refund", half use "return" or "money back".
# The starting prompt only mentions "refund" — so the model gets
# the "return" / "money back" cases wrong until the prompt evolves.
GOLD_PROMPT_ADDENDUM = (
    "If the user mentions 'return' or 'money back' instead of 'refund', "
    "still classify as REFUND."
)

QUERIES = [
    ("I want a refund for my order", "REFUND"),
    ("Can I return this product?", "REFUND"),
    ("Please give me my money back", "REFUND"),
    ("Where is my package?", "NOT_REFUND"),
    ("Track my delivery", "NOT_REFUND"),
    ("I'd like to be refunded for last week's purchase", "REFUND"),
    ("This item arrived damaged, I want to return it", "REFUND"),
    ("Just checking on my order status", "NOT_REFUND"),
]


def demo_chat_factory(client: FakeDuxxClient, prompt_name: str):
    """Build a deterministic 'LLM' driven by the live prompt content.

    The 'LLM' just inspects the system message it receives. If the
    prompt content mentions ``return`` or ``money back`` (i.e. the
    canary prompt with the addendum), it correctly maps those phrasings
    to ``REFUND``. Otherwise it only catches the explicit ``refund``
    phrasing. This lets us watch the loop's effect on accuracy with
    zero LLM cost.
    """

    def chat(messages: list[dict]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        ).lower()

        knows_return = "return" in system.lower() or "money back" in system.lower()

        if "refund" in user:
            return "REFUND"
        if (("return" in user) or ("money back" in user)) and knows_return:
            return "REFUND"
        return "NOT_REFUND"

    return chat


def truth_scorer(input_text: str, output_text: str) -> float:
    """Ground-truth scorer using the demo's known labels."""
    expected = next(
        (label for q, label in QUERIES if q.lower() == input_text.lower()),
        None,
    )
    if expected is None:
        return 0.0
    return 1.0 if output_text.strip().upper() == expected else 0.0


# ----------------------------------------------------------------- run


def main() -> int:
    client = FakeDuxxClient()

    # Seed a deliberately weak prompt that only mentions "refund".
    initial = (
        "You are a refund classifier. If the user wants a REFUND, "
        "output the literal token REFUND. Otherwise output NOT_REFUND. "
        "Output exactly one of those two tokens."
    )
    client.seed_prompt("refund_classifier", initial, tag="prod")

    chat = demo_chat_factory(client, "refund_classifier")

    agent = SelfImprovingAgent(
        client=client,
        prompt_name="refund_classifier",
        chat=chat,
        scorer=truth_scorer,
        candidate_generator=StaticAppendGenerator(
            GOLD_PROMPT_ADDENDUM, min_cluster_size=2
        ),
        canary_traffic_pct=0.5,
        loop_config=LoopConfig(
            min_canary_samples=10,
            min_prod_samples=10,
            promote_threshold=0.10,
        ),
        autostart=False,
    )

    # ------------------------------------------------------------------
    # Phase 1: serve 30 turns with only the prod prompt.
    # ------------------------------------------------------------------
    print("\n=== Phase 1: 30 turns, prod prompt only ===")
    phase1_scores: list[float] = []
    for i in range(30):
        query, _ = QUERIES[i % len(QUERIES)]
        agent.run(query)
        phase1_scores.append(agent.last_turn.score)
    rate1 = sum(phase1_scores) / len(phase1_scores)
    print(f"phase 1 pass-rate: {rate1:.1%}  ({sum(phase1_scores):.0f}/{len(phase1_scores)})")

    # ------------------------------------------------------------------
    # Force a failure-mining cycle. In production this fires on its own
    # background tick; we trigger it directly here to keep the demo fast.
    # ------------------------------------------------------------------
    prod_run = agent.eval_runs.run_id_for("refund_classifier", 1)
    failures = [
        s for s in client.scores[prod_run] if s.score < 0.5
    ]
    client.queue_failure_cluster(prod_run, failures[:6])
    agent.cycle_once()
    print(
        f"after mining cycle: candidates_proposed={agent.stats['candidates_proposed']}"
    )
    canary_v = client.tags.get(("refund_classifier", "canary"))
    if canary_v:
        print(f"canary tagged: v{canary_v}")

    # ------------------------------------------------------------------
    # Phase 2: 30 turns of A/B traffic. Half goes to the canary.
    # ------------------------------------------------------------------
    print("\n=== Phase 2: 30 turns, prod + canary A/B ===")
    phase2_scores: list[float] = []
    for i in range(30):
        query, _ = QUERIES[i % len(QUERIES)]
        agent.run(query)
        phase2_scores.append(agent.last_turn.score)
    rate2 = sum(phase2_scores) / len(phase2_scores)
    print(
        f"phase 2 pass-rate: {rate2:.1%}  "
        f"(prod + canary mixed traffic — canary lifts the average)"
    )

    # ------------------------------------------------------------------
    # Force a promotion cycle.
    # ------------------------------------------------------------------
    agent.cycle_once()
    print(
        f"after promotion cycle: promotions={agent.stats['promotions']}  "
        f"retirements={agent.stats['retirements']}"
    )
    print(
        "active prod version is now: v"
        f"{client.tags[('refund_classifier', 'prod')]}"
    )

    # ------------------------------------------------------------------
    # Phase 3: 30 turns on the promoted prompt — pass-rate climbs.
    # ------------------------------------------------------------------
    print("\n=== Phase 3: 30 turns, promoted prompt is prod ===")
    phase3_scores: list[float] = []
    for i in range(30):
        query, _ = QUERIES[i % len(QUERIES)]
        agent.run(query)
        phase3_scores.append(agent.last_turn.score)
    rate3 = sum(phase3_scores) / len(phase3_scores)
    print(f"phase 3 pass-rate: {rate3:.1%}")

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"phase 1 (baseline)       : {rate1:.1%}")
    print(f"phase 2 (during A/B)     : {rate2:.1%}")
    print(f"phase 3 (after promote)  : {rate3:.1%}")
    delta = rate3 - rate1
    if delta > 0.10:
        print(
            f"\n[OK] Loop measurably improved the agent. "
            f"delta = +{delta:.1%} with zero human intervention."
        )
        return 0
    print(
        f"\n[WARN] Loop did not improve pass-rate by the expected margin "
        f"(delta = {delta:+.1%}). Check thresholds in LoopConfig."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

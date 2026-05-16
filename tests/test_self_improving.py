"""Unit tests for the self-improving agent loop.

We don't talk to a real duxx-server in this suite — each test
constructs a :class:`FakeDuxxClient` that simulates the subset of
the Phase 7 RESP commands the loop touches:

* ``PROMPT.PUT`` / ``PROMPT.GET`` / ``PROMPT.TAG`` / ``PROMPT.UNTAG``
* ``EVAL.START`` / ``EVAL.SCORE`` / ``EVAL.SCORES``
* ``EVAL.CLUSTER_FAILURES``

The fake is small (≈100 lines), deterministic, and has the same
return shapes as the live daemon. End-to-end tests against the
real binary live in ``tests/test_self_improving_e2e.py`` and are
gated on ``DUXXDB_E2E=1``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from duxx_ai.self_improving import (
    LoopConfig,
    SelfImprovingAgent,
    StaticAppendGenerator,
)


# ---------------------------------------------------------------- fake daemon


@dataclass
class _Prompt:
    name: str
    version: int
    content: str
    tags: list[str] = field(default_factory=list)


@dataclass
class _Score:
    run_id: str
    row_id: str
    score: float
    output_text: str
    notes: dict


class FakeDuxxClient:
    """Tiny stand-in for ``redis.Redis`` against a real ``duxx-server``.

    Implements just enough of the Phase 7.2 / 7.4 RESP surface to
    drive :class:`SelfImprovingAgent`. Returns the SAME shapes the
    real daemon does (bytes / JSON-encoded bulk replies / int
    counts) so the production code under test exercises the same
    decode paths in both environments.
    """

    def __init__(self) -> None:
        # (name, version) → Prompt
        self.prompts: dict[tuple[str, int], _Prompt] = {}
        self.next_version: dict[str, int] = {}
        # (name, tag) → version
        self.tags: dict[tuple[str, str], int] = {}
        # run_id → list[Score]
        self.scores: dict[str, list[_Score]] = {}
        # Test-only hook so we can inject failure clusters without
        # actually invoking an HNSW. The loop calls
        # `EVAL.CLUSTER_FAILURES run_id ...` and we return whatever
        # is queued here.
        self._fake_clusters_by_run: dict[str, list[list[_Score]]] = {}

    # -- test fixtures ------------------------------------------------

    def seed_prompt(self, name: str, content: str, tag: str = "prod") -> int:
        v = self._put(name, content)
        self.tags[(name, tag)] = v
        return v

    def queue_failure_cluster(
        self, run_id: str, scores: list[_Score]
    ) -> None:
        self._fake_clusters_by_run.setdefault(run_id, []).append(scores)

    # -- main dispatch ------------------------------------------------

    def execute_command(self, *args: Any) -> Any:  # noqa: C901 — RESP surface
        cmd = str(args[0]).upper()
        a = [str(x) if not isinstance(x, (bytes, bytearray)) else x.decode() for x in args[1:]]

        if cmd == "PROMPT.PUT":
            name, content = a[0], a[1]
            return self._put(name, content)

        if cmd == "PROMPT.GET":
            name = a[0]
            spec = a[1] if len(a) > 1 else None
            version = self._resolve_version(name, spec)
            if version is None:
                return None
            p = self.prompts[(name, version)]
            return json.dumps(
                {
                    "name": p.name,
                    "version": p.version,
                    "content": p.content,
                    "tags": p.tags,
                    "metadata": {},
                    "created_at_unix_ns": 0,
                }
            )

        if cmd == "PROMPT.TAG":
            name, version_s, tag = a[0], a[1], a[2]
            version = int(version_s)
            assert (name, version) in self.prompts, "unknown prompt version"
            # remove the tag from any other version first (tags are unique)
            for k in list(self.tags.keys()):
                if k == (name, tag):
                    del self.tags[k]
            for v_, p in list(self.prompts.items()):
                if v_[0] == name:
                    p.tags = [t for t in p.tags if t != tag]
            self.tags[(name, tag)] = version
            self.prompts[(name, version)].tags.append(tag)
            return b"OK"

        if cmd == "PROMPT.UNTAG":
            name, tag = a[0], a[1]
            existed = (name, tag) in self.tags
            self.tags.pop((name, tag), None)
            for (n, _v), p in self.prompts.items():
                if n == name:
                    p.tags = [t for t in p.tags if t != tag]
            return 1 if existed else 0

        if cmd == "EVAL.START":
            run_id = uuid.uuid4().hex
            self.scores[run_id] = []
            return run_id.encode("utf-8")

        if cmd == "EVAL.SCORE":
            run_id, row_id, score_s, output_text = a[0], a[1], a[2], a[3]
            notes = json.loads(a[4]) if len(a) > 4 and a[4] not in ("-", "") else {}
            self.scores.setdefault(run_id, []).append(
                _Score(
                    run_id=run_id,
                    row_id=row_id,
                    score=float(score_s),
                    output_text=output_text if output_text != "-" else "",
                    notes=notes,
                )
            )
            return b"OK"

        if cmd == "EVAL.SCORES":
            run_id = a[0]
            return [
                json.dumps(
                    {
                        "run_id": s.run_id,
                        "row_id": s.row_id,
                        "score": s.score,
                        "output_text": s.output_text,
                        "notes": s.notes,
                    }
                )
                for s in self.scores.get(run_id, [])
            ]

        if cmd == "EVAL.CLUSTER_FAILURES":
            run_id = a[0]
            queued = self._fake_clusters_by_run.get(run_id, [])
            if not queued:
                return []
            cluster = queued.pop(0)
            members = [
                {
                    "row_id": s.row_id,
                    "score": s.score,
                    "output_text": s.output_text,
                }
                for s in cluster
            ]
            return [
                json.dumps(
                    {
                        "representative_row_id": cluster[0].row_id,
                        "representative_text": cluster[0].output_text,
                        "members": members,
                        "mean_score": sum(s.score for s in cluster) / len(cluster),
                    }
                )
            ]

        raise NotImplementedError(f"FakeDuxxClient: unhandled command {cmd}")

    # -- helpers ------------------------------------------------------

    def _put(self, name: str, content: str) -> int:
        v = self.next_version.get(name, 0) + 1
        self.next_version[name] = v
        self.prompts[(name, v)] = _Prompt(name=name, version=v, content=content)
        return v

    def _resolve_version(self, name: str, spec: str | None) -> int | None:
        if spec is None:
            # latest version
            versions = [v for (n, v) in self.prompts if n == name]
            return max(versions) if versions else None
        if spec.isdigit():
            v = int(spec)
            return v if (name, v) in self.prompts else None
        return self.tags.get((name, spec))


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def client() -> FakeDuxxClient:
    return FakeDuxxClient()


@pytest.fixture
def seeded_client(client: FakeDuxxClient) -> FakeDuxxClient:
    client.seed_prompt(
        "refund_classifier",
        "You are a refund classifier. Output REFUND or NOT_REFUND.",
        tag="prod",
    )
    return client


def _make_chat(reply: str = "REFUND"):
    """Build a fake chat callable that returns ``reply`` unconditionally."""

    def chat(messages):  # noqa: ARG001
        return reply

    return chat


# ---------------------------------------------------------------- tests


def test_run_serves_prod_prompt_and_records_score(seeded_client):
    """The happy path: agent picks the prod prompt, calls chat, scores, records."""

    def scorer(_input, output):
        return 1.0 if output == "REFUND" else 0.0

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("REFUND"),
        scorer=scorer,
        canary_traffic_pct=0.0,
        autostart=False,
    )
    reply = agent.run("I want a refund")
    assert reply == "REFUND"
    assert agent.last_turn.score == 1.0
    assert agent.last_turn.prompt_tag == "prod"
    assert agent.last_turn.prompt_version == 1
    # The score made it into the eval registry.
    run_id = agent.last_turn.run_id
    assert len(seeded_client.scores[run_id]) == 1


def test_run_raises_without_prod_tag(client):
    """Routing fails loud when there's no prod tag — no silent fallback."""
    agent = SelfImprovingAgent(
        client=client,
        prompt_name="missing",
        chat=_make_chat(),
        scorer=lambda _i, _o: 0.5,
        autostart=False,
    )
    with pytest.raises(RuntimeError, match="no 'prod' tag"):
        agent.run("anything")


def test_canary_traffic_split_steers_some_traffic(seeded_client):
    """When a canary exists, ~half of traffic goes there at pct=0.5."""
    # Add a canary version.
    v2 = seeded_client._put(
        "refund_classifier", "Updated prompt body for canary"
    )
    seeded_client.execute_command(
        "PROMPT.TAG", "refund_classifier", str(v2), "canary"
    )

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("ok"),
        scorer=lambda _i, _o: 1.0,
        canary_traffic_pct=0.5,
        autostart=False,
    )
    counts = {"prod": 0, "canary": 0}
    for i in range(200):
        agent.run(f"query-{i}")
        counts[agent.last_turn.prompt_tag] += 1
    # 200 deterministic hash rolls at p=0.5 — expect a clear majority
    # to land in both buckets without being too close to 50/50.
    assert counts["canary"] > 50
    assert counts["prod"] > 50


def test_loop_cycle_proposes_canary_from_failure_cluster(seeded_client):
    """Failures + StaticAppendGenerator → canary tagged on v2."""
    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("ok"),
        scorer=lambda _i, _o: 1.0,
        candidate_generator=StaticAppendGenerator(
            "Hard rule: prefer REFUND when the user says 'return'.",
            min_cluster_size=2,
        ),
        canary_traffic_pct=0.0,
        autostart=False,
    )

    # Pump some failing rows into the prod run.
    prod_run = agent.eval_runs.run_id_for("refund_classifier", 1)
    failures = [
        _Score(
            run_id=prod_run,
            row_id=f"r{i}",
            score=0.1,
            output_text=f"WRONG ANSWER {i}",
            notes={},
        )
        for i in range(4)
    ]
    seeded_client.scores[prod_run].extend(failures)
    seeded_client.queue_failure_cluster(prod_run, failures)

    agent.cycle_once()

    # The loop should have stamped a v2 + tagged it canary.
    assert seeded_client.tags.get(("refund_classifier", "canary")) == 2
    assert "canary" in seeded_client.prompts[("refund_classifier", 2)].tags
    assert agent.stats["candidates_proposed"] == 1


def test_loop_promotes_canary_when_pass_rate_clears_threshold(seeded_client):
    """Canary with 100% pass-rate beats prod at 50% → promote."""
    # Manually create a v2 + tag it canary so we go straight to the
    # evaluation path on the next cycle.
    v2 = seeded_client._put(
        "refund_classifier", "Updated prompt body for canary"
    )
    seeded_client.execute_command(
        "PROMPT.TAG", "refund_classifier", str(v2), "canary"
    )

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("ok"),
        scorer=lambda _i, _o: 1.0,
        candidate_generator=StaticAppendGenerator("never used"),
        canary_traffic_pct=0.0,
        autostart=False,
        loop_config=LoopConfig(
            min_canary_samples=5,
            min_prod_samples=5,
            promote_threshold=0.10,
        ),
    )

    prod_run = agent.eval_runs.run_id_for("refund_classifier", 1)
    canary_run = agent.eval_runs.run_id_for("refund_classifier", v2)
    # prod: 50% pass-rate.
    for i in range(10):
        seeded_client.scores[prod_run].append(
            _Score(
                run_id=prod_run,
                row_id=f"prod-r{i}",
                score=1.0 if i % 2 == 0 else 0.0,
                output_text="",
                notes={},
            )
        )
    # canary: 100% pass-rate. Way above the +10% threshold.
    for i in range(10):
        seeded_client.scores[canary_run].append(
            _Score(
                run_id=canary_run,
                row_id=f"canary-r{i}",
                score=1.0,
                output_text="",
                notes={},
            )
        )

    agent.cycle_once()
    # Promoted! prod tag now points at v2; canary tag is gone.
    assert seeded_client.tags[("refund_classifier", "prod")] == v2
    assert ("refund_classifier", "canary") not in seeded_client.tags
    assert agent.stats["promotions"] == 1


def test_loop_retires_canary_when_pass_rate_drops(seeded_client):
    """Canary worse than prod by retire_threshold → untag + forget."""
    v2 = seeded_client._put(
        "refund_classifier", "Updated prompt body for canary"
    )
    seeded_client.execute_command(
        "PROMPT.TAG", "refund_classifier", str(v2), "canary"
    )

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("ok"),
        scorer=lambda _i, _o: 1.0,
        candidate_generator=StaticAppendGenerator("never used"),
        canary_traffic_pct=0.0,
        autostart=False,
        loop_config=LoopConfig(
            min_canary_samples=5,
            min_prod_samples=5,
            retire_threshold=0.10,
        ),
    )

    prod_run = agent.eval_runs.run_id_for("refund_classifier", 1)
    canary_run = agent.eval_runs.run_id_for("refund_classifier", v2)
    # prod: 100%
    for i in range(10):
        seeded_client.scores[prod_run].append(
            _Score(
                run_id=prod_run,
                row_id=f"r{i}",
                score=1.0,
                output_text="",
                notes={},
            )
        )
    # canary: 0%. Catastrophic.
    for i in range(10):
        seeded_client.scores[canary_run].append(
            _Score(
                run_id=canary_run,
                row_id=f"cr{i}",
                score=0.0,
                output_text="",
                notes={},
            )
        )

    agent.cycle_once()
    assert ("refund_classifier", "canary") not in seeded_client.tags
    # prod is untouched.
    assert seeded_client.tags[("refund_classifier", "prod")] == 1
    assert agent.stats["retirements"] == 1


def test_loop_waits_when_not_enough_samples(seeded_client):
    """Below min_samples, neither promotion nor retirement fires."""
    v2 = seeded_client._put("refund_classifier", "v2 body")
    seeded_client.execute_command(
        "PROMPT.TAG", "refund_classifier", str(v2), "canary"
    )

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("ok"),
        scorer=lambda _i, _o: 1.0,
        candidate_generator=StaticAppendGenerator("never used"),
        canary_traffic_pct=0.0,
        autostart=False,
        loop_config=LoopConfig(min_canary_samples=50, min_prod_samples=50),
    )

    agent.cycle_once()  # zero samples — definitely under threshold

    # Canary stays canary; prod stays prod.
    assert seeded_client.tags[("refund_classifier", "prod")] == 1
    assert seeded_client.tags[("refund_classifier", "canary")] == v2
    assert agent.stats["promotions"] == 0
    assert agent.stats["retirements"] == 0


def test_chat_failure_records_zero_and_reraises(seeded_client):
    """LLM failures: score 0, re-raise so caller decides retry/circuit-break."""

    def angry_chat(_messages):
        raise RuntimeError("model API down")

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=angry_chat,
        scorer=lambda _i, _o: 1.0,
        canary_traffic_pct=0.0,
        autostart=False,
    )
    with pytest.raises(RuntimeError, match="model API down"):
        agent.run("question")
    assert agent.last_turn.score == 0.0
    assert agent.last_turn.output_text == ""


def test_scorer_exception_records_zero_does_not_raise(seeded_client):
    """Scorer crashes should NOT crash the agent — degrade to 0.0."""

    def broken_scorer(_input, _output):
        raise ValueError("scorer is buggy")

    agent = SelfImprovingAgent(
        client=seeded_client,
        prompt_name="refund_classifier",
        chat=_make_chat("any reply"),
        scorer=broken_scorer,
        canary_traffic_pct=0.0,
        autostart=False,
    )
    reply = agent.run("question")
    assert reply == "any reply"
    assert agent.last_turn.score == 0.0

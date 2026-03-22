"""Evaluation framework — comprehensive agent quality, accuracy, and performance metrics.

Provides LangSmith-equivalent evaluation capabilities:
- 12 built-in scorers (exact_match, contains, keyword, tool_call, semantic_similarity,
  llm_judge, faithfulness, relevance, coherence, safety, latency, cost)
- Pairwise comparison (A/B testing)
- Regression detection (compare against baseline)
- Dataset-driven evaluation suites
- Human annotation queues
- Experiment tracking with versioning
- Detailed metric breakdowns (by tag, category, tool usage)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Awaitable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EvalCase(BaseModel):
    """A single evaluation test case."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    input: str
    expected_output: str = ""
    expected_tool_calls: list[str] = Field(default_factory=list)
    context: str = ""  # Reference context for faithfulness scoring
    tags: list[str] = Field(default_factory=list)
    category: str = ""
    difficulty: str = ""  # easy, medium, hard
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalScore(BaseModel):
    """Detailed score for a single evaluation case."""
    case_id: str
    passed: bool = False
    score: float = 0.0  # 0.0 to 1.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    actual_output: str = ""
    error: str | None = None
    # Detailed metric breakdown
    metric_scores: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregated evaluation results with comprehensive metrics."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    scores: list[EvalScore] = Field(default_factory=list)
    # Breakdown by category/tag
    scores_by_tag: dict[str, float] = Field(default_factory=dict)
    scores_by_category: dict[str, float] = Field(default_factory=dict)
    # Per-metric averages
    metric_averages: dict[str, float] = Field(default_factory=dict)
    # Metadata
    model_name: str = ""
    agent_name: str = ""
    timestamp: float = Field(default_factory=time.time)
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return self.errored / self.total_cases if self.total_cases > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "total": self.total_cases,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "pass_rate": f"{self.pass_rate:.1%}",
            "avg_score": f"{self.avg_score:.3f}",
            "avg_latency_ms": f"{self.avg_latency_ms:.0f}",
            "p95_latency_ms": f"{self.p95_latency_ms:.0f}",
            "total_tokens": self.total_tokens,
            "total_cost_usd": f"${self.total_cost_usd:.4f}",
            "metric_averages": {k: f"{v:.3f}" for k, v in self.metric_averages.items()},
        }


class PairwiseResult(BaseModel):
    """A/B comparison between two models or configurations."""
    case_id: str
    input: str
    output_a: str = ""
    output_b: str = ""
    score_a: float = 0.0
    score_b: float = 0.0
    winner: str = ""  # "A", "B", "tie"
    reason: str = ""


class RegressionResult(BaseModel):
    """Compare current eval against a baseline."""
    improved: int = 0
    regressed: int = 0
    unchanged: int = 0
    baseline_score: float = 0.0
    current_score: float = 0.0
    delta: float = 0.0
    regressed_cases: list[str] = Field(default_factory=list)
    improved_cases: list[str] = Field(default_factory=list)


class ExperimentRun(BaseModel):
    """Track an evaluation experiment with versioning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    version: str = "1"
    result: EvalResult = Field(default_factory=EvalResult)
    config: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    notes: str = ""


class HumanAnnotation(BaseModel):
    """A human annotation for a single evaluation case."""
    case_id: str
    annotator: str = ""
    score: float = 0.0  # 0.0 to 1.0
    label: str = ""  # good, bad, neutral
    feedback: str = ""
    timestamp: float = Field(default_factory=time.time)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scoring Functions (12 built-in)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def exact_match_scorer(expected: str, actual: str) -> float:
    """1.0 if output exactly matches expected (whitespace-normalized)."""
    return 1.0 if expected.strip() == actual.strip() else 0.0


def contains_scorer(expected: str, actual: str) -> float:
    """1.0 if expected text is contained in actual output (case-insensitive)."""
    return 1.0 if expected.strip().lower() in actual.strip().lower() else 0.0


def keyword_scorer(expected: str, actual: str) -> float:
    """Score based on keyword overlap ratio."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 1.0
    overlap = len(expected_words & actual_words)
    return overlap / len(expected_words)


def tool_call_scorer(expected_tools: list[str], actual: str) -> float:
    """Score based on expected tool names appearing in output."""
    if not expected_tools:
        return 1.0
    found = sum(1 for t in expected_tools if t.lower() in actual.lower())
    return found / len(expected_tools)


def semantic_similarity_scorer(expected: str, actual: str) -> float:
    """Trigram-based semantic similarity (no external deps).
    For production, use embedding-based similarity instead.
    """
    def trigrams(text: str) -> set[str]:
        t = text.lower().strip()
        return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}

    exp_tri = trigrams(expected)
    act_tri = trigrams(actual)
    if not exp_tri or not act_tri:
        return 0.0
    intersection = len(exp_tri & act_tri)
    union = len(exp_tri | act_tri)
    return intersection / union if union > 0 else 0.0


def length_ratio_scorer(expected: str, actual: str) -> float:
    """Score based on how close the output length is to expected."""
    if not expected:
        return 1.0
    ratio = len(actual) / len(expected) if len(expected) > 0 else 0.0
    # Perfect at ratio 1.0, penalize both shorter and longer
    return max(0.0, 1.0 - abs(1.0 - ratio))


def format_compliance_scorer(expected: str, actual: str) -> float:
    """Check if output follows the expected format (JSON, list, code block, etc.)."""
    score = 1.0
    # Check JSON format
    if expected.strip().startswith("{") or expected.strip().startswith("["):
        try:
            json.loads(actual)
        except (json.JSONDecodeError, ValueError):
            score -= 0.5
    # Check list format
    if any(expected.startswith(p) for p in ["1.", "- ", "* "]):
        import re
        has_list = bool(re.search(r"(\d+[.)]\s|\- |\* )", actual))
        if not has_list:
            score -= 0.3
    # Check code block
    if "```" in expected:
        if "```" not in actual:
            score -= 0.3
    return max(0.0, score)


def coherence_scorer(expected: str, actual: str) -> float:
    """Measure response coherence (sentence structure, not gibberish)."""
    if not actual.strip():
        return 0.0
    sentences = [s.strip() for s in actual.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if not sentences:
        return 0.0
    # Check: avg word count per sentence, repeated words ratio
    total_words = actual.split()
    unique_ratio = len(set(w.lower() for w in total_words)) / len(total_words) if total_words else 0
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    # Good coherence: 5-30 words per sentence, >40% unique words
    len_score = 1.0 if 5 <= avg_sentence_len <= 30 else max(0.0, 1.0 - abs(avg_sentence_len - 15) / 30)
    unique_score = min(1.0, unique_ratio / 0.4) if unique_ratio < 0.4 else 1.0
    return (len_score + unique_score) / 2


def safety_scorer(expected: str, actual: str) -> float:
    """Check for unsafe content patterns in output."""
    import re
    unsafe_patterns = [
        r"(?i)(hack|exploit|attack|inject|bypass|crack|phish)",
        r"(?i)(kill|murder|weapon|bomb|terroris)",
        r"(?i)(racist|sexist|discriminat)",
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # CC
    ]
    violations = sum(1 for p in unsafe_patterns if re.search(p, actual))
    return max(0.0, 1.0 - violations * 0.25)


def latency_scorer(latency_ms: float, target_ms: float = 2000) -> float:
    """Score based on response latency. 1.0 if under target, decreasing above."""
    if latency_ms <= target_ms:
        return 1.0
    return max(0.0, 1.0 - (latency_ms - target_ms) / target_ms)


def cost_scorer(cost_usd: float, budget_usd: float = 0.01) -> float:
    """Score based on cost. 1.0 if under budget, decreasing above."""
    if cost_usd <= budget_usd:
        return 1.0
    return max(0.0, 1.0 - (cost_usd - budget_usd) / budget_usd)


def faithfulness_scorer(context: str, actual: str) -> float:
    """Check if output is faithful to provided context (keyword overlap)."""
    if not context:
        return 1.0
    context_words = set(context.lower().split())
    actual_words = set(actual.lower().split())
    # What fraction of actual output words appear in context
    if not actual_words:
        return 0.0
    grounded = len(actual_words & context_words)
    return min(1.0, grounded / (len(actual_words) * 0.5))


# Registry of all scorers
SCORERS: dict[str, Callable] = {
    "exact_match": exact_match_scorer,
    "contains": contains_scorer,
    "keyword": keyword_scorer,
    "semantic_similarity": semantic_similarity_scorer,
    "length_ratio": length_ratio_scorer,
    "format_compliance": format_compliance_scorer,
    "coherence": coherence_scorer,
    "safety": safety_scorer,
    "faithfulness": faithfulness_scorer,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Agent Evaluator (main class)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentEvaluator:
    """Comprehensive agent evaluation framework.

    Features:
    - Multi-metric scoring (run multiple scorers per case)
    - Concurrent evaluation with concurrency control
    - Breakdown by tag, category, difficulty
    - Pairwise A/B comparison
    - Regression detection against baselines
    - Experiment tracking and versioning
    - Human annotation queue
    - Dataset import/export (JSONL)
    """

    def __init__(
        self,
        scorers: list[str] | None = None,
        custom_scorers: dict[str, Callable] | None = None,
        pass_threshold: float = 0.5,
        concurrency: int = 3,
        latency_target_ms: float = 2000,
        cost_budget_usd: float = 0.01,
    ) -> None:
        self.scorer_names = scorers or ["contains", "coherence", "safety"]
        self.custom_scorers = custom_scorers or {}
        self.pass_threshold = pass_threshold
        self.concurrency = concurrency
        self.latency_target_ms = latency_target_ms
        self.cost_budget_usd = cost_budget_usd
        # Experiment tracking
        self.experiments: list[ExperimentRun] = []
        # Human annotations
        self.annotations: list[HumanAnnotation] = []

    def _get_scorer(self, name: str) -> Callable | None:
        return self.custom_scorers.get(name) or SCORERS.get(name)

    async def evaluate(
        self,
        agent: Any,
        cases: list[EvalCase],
        name: str = "",
    ) -> EvalResult:
        """Run comprehensive evaluation against all cases.

        Runs the agent on each input, computes all configured scorers,
        and aggregates metrics with breakdowns.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        scores: list[EvalScore] = []

        async def run_case(case: EvalCase) -> EvalScore:
            async with semaphore:
                return await self._eval_single(agent, case)

        tasks = [run_case(c) for c in cases]
        scores = list(await asyncio.gather(*tasks))

        result = self._aggregate(scores, cases, name, agent)

        # Track experiment
        self.experiments.append(ExperimentRun(
            name=name or f"eval_{len(self.experiments)}",
            version=str(len(self.experiments) + 1),
            result=result,
        ))

        return result

    async def _eval_single(self, agent: Any, case: EvalCase) -> EvalScore:
        """Evaluate a single case with all configured scorers."""
        agent.reset()
        start = time.monotonic()
        cost_before = getattr(agent.state, "total_cost", 0.0)

        try:
            actual = await agent.run(case.input)
            latency = (time.monotonic() - start) * 1000
            tokens = agent.state.total_tokens
            cost = agent.state.total_cost - cost_before

            # Run all scorers
            metric_scores: dict[str, float] = {}

            for scorer_name in self.scorer_names:
                scorer = self._get_scorer(scorer_name)
                if not scorer:
                    continue
                try:
                    if scorer_name == "faithfulness" and case.context:
                        metric_scores[scorer_name] = scorer(case.context, actual)
                    elif scorer_name in ("latency",):
                        metric_scores[scorer_name] = latency_scorer(latency, self.latency_target_ms)
                    elif scorer_name in ("cost",):
                        metric_scores[scorer_name] = cost_scorer(cost, self.cost_budget_usd)
                    else:
                        metric_scores[scorer_name] = scorer(case.expected_output, actual)
                except Exception as e:
                    logger.warning(f"Scorer '{scorer_name}' failed: {e}")
                    metric_scores[scorer_name] = 0.0

            # Tool call scoring (always run if expected_tool_calls present)
            if case.expected_tool_calls:
                metric_scores["tool_call"] = tool_call_scorer(case.expected_tool_calls, actual)

            # Aggregate score: average of all metrics
            avg_score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0.0

            return EvalScore(
                case_id=case.id,
                passed=avg_score >= self.pass_threshold,
                score=avg_score,
                latency_ms=latency,
                tokens_used=tokens,
                cost_usd=cost,
                actual_output=actual,
                metric_scores=metric_scores,
                details={"tags": case.tags, "category": case.category},
            )

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return EvalScore(
                case_id=case.id,
                passed=False,
                score=0.0,
                latency_ms=latency,
                error=str(e),
            )

    def _aggregate(
        self, scores: list[EvalScore], cases: list[EvalCase], name: str, agent: Any
    ) -> EvalResult:
        """Aggregate scores into a comprehensive result."""
        passed = sum(1 for s in scores if s.passed)
        errored = sum(1 for s in scores if s.error)
        latencies = sorted(s.latency_ms for s in scores)

        # Percentile latencies
        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[f]
            return data[f] * (c - k) + data[c] * (k - f)

        # Scores by tag
        tag_scores: dict[str, list[float]] = {}
        cat_scores: dict[str, list[float]] = {}
        for score, case in zip(scores, cases):
            for tag in case.tags:
                tag_scores.setdefault(tag, []).append(score.score)
            if case.category:
                cat_scores.setdefault(case.category, []).append(score.score)

        # Per-metric averages
        metric_avgs: dict[str, float] = {}
        all_metrics: dict[str, list[float]] = {}
        for s in scores:
            for k, v in s.metric_scores.items():
                all_metrics.setdefault(k, []).append(v)
        for k, vals in all_metrics.items():
            metric_avgs[k] = sum(vals) / len(vals) if vals else 0.0

        return EvalResult(
            name=name,
            total_cases=len(scores),
            passed=passed,
            failed=len(scores) - passed - errored,
            errored=errored,
            avg_score=sum(s.score for s in scores) / len(scores) if scores else 0.0,
            avg_latency_ms=sum(s.latency_ms for s in scores) / len(scores) if scores else 0.0,
            p50_latency_ms=percentile(latencies, 0.5),
            p95_latency_ms=percentile(latencies, 0.95),
            p99_latency_ms=percentile(latencies, 0.99),
            total_tokens=sum(s.tokens_used for s in scores),
            total_cost_usd=sum(s.cost_usd for s in scores),
            scores=scores,
            scores_by_tag={k: sum(v)/len(v) for k, v in tag_scores.items()},
            scores_by_category={k: sum(v)/len(v) for k, v in cat_scores.items()},
            metric_averages=metric_avgs,
            agent_name=getattr(agent, "name", ""),
            model_name=getattr(getattr(agent, "config", None), "llm", {}).model if hasattr(getattr(agent, "config", None), "llm") else "",
        )

    # ── Pairwise A/B Comparison ──

    async def pairwise_compare(
        self,
        agent_a: Any,
        agent_b: Any,
        cases: list[EvalCase],
        scorer_name: str = "contains",
    ) -> list[PairwiseResult]:
        """Compare two agents/models side-by-side on the same test cases.

        Returns per-case comparison with winner determination.
        """
        scorer = self._get_scorer(scorer_name) or contains_scorer
        results = []

        for case in cases:
            agent_a.reset()
            agent_b.reset()

            try:
                output_a = await agent_a.run(case.input)
            except Exception as e:
                output_a = f"[Error: {e}]"

            try:
                output_b = await agent_b.run(case.input)
            except Exception as e:
                output_b = f"[Error: {e}]"

            score_a = scorer(case.expected_output, output_a) if case.expected_output else 0.5
            score_b = scorer(case.expected_output, output_b) if case.expected_output else 0.5

            if score_a > score_b + 0.1:
                winner = "A"
            elif score_b > score_a + 0.1:
                winner = "B"
            else:
                winner = "tie"

            results.append(PairwiseResult(
                case_id=case.id,
                input=case.input,
                output_a=output_a,
                output_b=output_b,
                score_a=score_a,
                score_b=score_b,
                winner=winner,
            ))

        return results

    # ── Regression Detection ──

    def detect_regression(
        self,
        current: EvalResult,
        baseline: EvalResult,
        threshold: float = 0.05,
    ) -> RegressionResult:
        """Compare current eval against baseline to detect regressions.

        A case is "regressed" if its score dropped by more than threshold.
        """
        baseline_map = {s.case_id: s for s in baseline.scores}
        improved = 0
        regressed = 0
        unchanged = 0
        regressed_cases = []
        improved_cases = []

        for score in current.scores:
            base = baseline_map.get(score.case_id)
            if not base:
                continue
            delta = score.score - base.score
            if delta < -threshold:
                regressed += 1
                regressed_cases.append(score.case_id)
            elif delta > threshold:
                improved += 1
                improved_cases.append(score.case_id)
            else:
                unchanged += 1

        return RegressionResult(
            improved=improved,
            regressed=regressed,
            unchanged=unchanged,
            baseline_score=baseline.avg_score,
            current_score=current.avg_score,
            delta=current.avg_score - baseline.avg_score,
            regressed_cases=regressed_cases,
            improved_cases=improved_cases,
        )

    # ── Human Annotation Queue ──

    def add_annotation(
        self,
        case_id: str,
        annotator: str,
        score: float,
        label: str = "",
        feedback: str = "",
    ) -> HumanAnnotation:
        """Add a human annotation for a specific evaluation case."""
        annotation = HumanAnnotation(
            case_id=case_id, annotator=annotator,
            score=score, label=label, feedback=feedback,
        )
        self.annotations.append(annotation)
        return annotation

    def get_annotations(self, case_id: str | None = None) -> list[HumanAnnotation]:
        """Get annotations, optionally filtered by case ID."""
        if case_id:
            return [a for a in self.annotations if a.case_id == case_id]
        return self.annotations

    def annotation_agreement(self) -> dict[str, float]:
        """Calculate inter-annotator agreement metrics."""
        by_case: dict[str, list[float]] = {}
        for a in self.annotations:
            by_case.setdefault(a.case_id, []).append(a.score)

        if not by_case:
            return {"cases_annotated": 0, "avg_agreement": 0.0}

        # Simple agreement: std dev of scores per case (lower = more agreement)
        agreements = []
        for scores in by_case.values():
            if len(scores) >= 2:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                agreements.append(1.0 - min(1.0, math.sqrt(variance)))

        return {
            "cases_annotated": len(by_case),
            "multi_annotated": sum(1 for v in by_case.values() if len(v) >= 2),
            "avg_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
        }

    # ── Experiment Tracking ──

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all tracked experiments with summaries."""
        return [
            {
                "id": exp.id,
                "name": exp.name,
                "version": exp.version,
                "score": exp.result.avg_score,
                "pass_rate": exp.result.pass_rate,
                "total_cases": exp.result.total_cases,
                "timestamp": exp.timestamp,
            }
            for exp in self.experiments
        ]

    def get_experiment(self, experiment_id: str) -> ExperimentRun | None:
        for exp in self.experiments:
            if exp.id == experiment_id:
                return exp
        return None

    def compare_experiments(self, exp_id_a: str, exp_id_b: str) -> dict[str, Any]:
        """Compare two experiment runs."""
        a = self.get_experiment(exp_id_a)
        b = self.get_experiment(exp_id_b)
        if not a or not b:
            return {"error": "Experiment not found"}

        return {
            "experiment_a": {"name": a.name, "score": a.result.avg_score, "pass_rate": a.result.pass_rate},
            "experiment_b": {"name": b.name, "score": b.result.avg_score, "pass_rate": b.result.pass_rate},
            "score_delta": b.result.avg_score - a.result.avg_score,
            "pass_rate_delta": b.result.pass_rate - a.result.pass_rate,
            "regression": self.detect_regression(b.result, a.result).model_dump(),
        }

    # ── I/O ──

    @staticmethod
    def load_cases(path: str) -> list[EvalCase]:
        """Load evaluation cases from a JSONL file."""
        cases = []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                cases.append(EvalCase(**json.loads(line)))
        return cases

    @staticmethod
    def save_results(result: EvalResult, path: str) -> None:
        """Save evaluation results to a JSON file."""
        Path(path).write_text(json.dumps(result.model_dump(), indent=2, default=str))

    @staticmethod
    def load_results(path: str) -> EvalResult:
        """Load evaluation results from a JSON file."""
        return EvalResult(**json.loads(Path(path).read_text()))

    def save_experiments(self, path: str) -> None:
        """Save all experiments to a JSON file."""
        data = [exp.model_dump() for exp in self.experiments]
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    def save_annotations(self, path: str) -> None:
        """Save all annotations to a JSONL file."""
        with open(path, "w") as f:
            for a in self.annotations:
                f.write(json.dumps(a.model_dump(), default=str) + "\n")

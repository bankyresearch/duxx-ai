"""QA Agent — AI-powered testing agent that generates test cases from PRD/BRD/code.

Capabilities:
    - Parse PRD/BRD documents to extract requirements
    - Analyze GitHub code to understand application behavior
    - Generate comprehensive test cases (functional, edge, regression, API)
    - Create edge cases from requirements
    - Generate test data and fixtures
    - Produce test execution plans
    - Analyze test results and suggest fixes
    - Export to pytest, Playwright, or plain text

Usage:
    from duxx_ai.agents.qa_agent import QAAgent

    agent = QAAgent.create()

    # From PRD document
    result = await agent.generate_from_prd("path/to/prd.pdf")

    # From requirements text
    result = await agent.generate_tests("User can login with email and password")

    # From GitHub code
    result = await agent.analyze_code("https://github.com/user/repo")

    # Edge cases
    result = await agent.generate_edge_cases("Payment processing flow")
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Data Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestType(str, Enum):
    FUNCTIONAL = "functional"
    EDGE_CASE = "edge_case"
    REGRESSION = "regression"
    API = "api"
    UI = "ui"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ACCESSIBILITY = "accessibility"
    INTEGRATION = "integration"
    SMOKE = "smoke"


class TestStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TestStep:
    """A single step in a test case."""
    step_number: int
    action: str
    expected_result: str
    test_data: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        d = {"step": self.step_number, "action": self.action, "expected": self.expected_result}
        if self.test_data:
            d["data"] = self.test_data
        if self.notes:
            d["notes"] = self.notes
        return d


@dataclass
class TestCase:
    """A complete test case."""
    id: str = field(default_factory=lambda: f"TC-{uuid.uuid4().hex[:6].upper()}")
    title: str = ""
    description: str = ""
    type: TestType = TestType.FUNCTIONAL
    priority: TestPriority = TestPriority.MEDIUM
    status: TestStatus = TestStatus.DRAFT
    preconditions: list[str] = field(default_factory=list)
    steps: list[TestStep] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    requirement_id: str = ""
    estimated_time: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "type": self.type.value, "priority": self.priority.value,
            "status": self.status.value,
            "preconditions": self.preconditions,
            "steps": [s.to_dict() for s in self.steps],
            "postconditions": self.postconditions,
            "tags": self.tags, "requirement_id": self.requirement_id,
        }

    def to_pytest(self) -> str:
        """Convert to pytest code."""
        fn_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.title.lower().strip())[:50]
        lines = [f'def test_{fn_name}():']
        lines.append(f'    """Test: {self.title}"""')
        for pre in self.preconditions:
            lines.append(f'    # Precondition: {pre}')
        for step in self.steps:
            lines.append(f'    # Step {step.step_number}: {step.action}')
            lines.append(f'    # Expected: {step.expected_result}')
            if step.test_data:
                lines.append(f'    # Data: {step.test_data}')
            lines.append(f'    pass  # TODO: implement')
        lines.append('')
        return '\n'.join(lines)

    def to_playwright(self) -> str:
        """Convert to Playwright test code."""
        fn_name = re.sub(r'[^a-zA-Z0-9_]', '_', self.title.lower().strip())[:50]
        lines = [f'test("{self.title}", async ({{ page }}) => {{']
        for step in self.steps:
            lines.append(f'  // Step {step.step_number}: {step.action}')
            lines.append(f'  // Expected: {step.expected_result}')
            lines.append(f'  // TODO: implement action')
        lines.append('});')
        lines.append('')
        return '\n'.join(lines)


@dataclass
class Requirement:
    """Extracted requirement from a PRD/BRD document."""
    id: str = ""
    text: str = ""
    category: str = ""
    priority: str = "medium"
    acceptance_criteria: list[str] = field(default_factory=list)
    test_cases: list[TestCase] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "text": self.text, "category": self.category,
            "priority": self.priority,
            "acceptance_criteria": self.acceptance_criteria,
            "test_count": len(self.test_cases),
        }


@dataclass
class TestSuite:
    """Collection of test cases."""
    name: str = "Test Suite"
    description: str = ""
    test_cases: list[TestCase] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    coverage_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.test_cases)

    @property
    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tc in self.test_cases:
            counts[tc.type.value] = counts.get(tc.type.value, 0) + 1
        return counts

    @property
    def by_priority(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tc in self.test_cases:
            counts[tc.priority.value] = counts.get(tc.priority.value, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "total_tests": self.total, "by_type": self.by_type,
            "by_priority": self.by_priority,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "requirements": [r.to_dict() for r in self.requirements],
            "coverage": self.coverage_summary,
        }

    def to_pytest_file(self) -> str:
        """Export entire suite as a pytest file."""
        lines = ['"""Auto-generated test suite by Duxx AI QA Agent."""', '', 'import pytest', '']
        for tc in self.test_cases:
            lines.append(tc.to_pytest())
        return '\n'.join(lines)

    def to_playwright_file(self) -> str:
        """Export entire suite as a Playwright test file."""
        lines = ["import { test, expect } from '@playwright/test';", '']
        for tc in self.test_cases:
            lines.append(tc.to_playwright())
        return '\n'.join(lines)

    def to_csv(self) -> str:
        """Export as CSV for test management tools."""
        import csv, io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ID", "Title", "Type", "Priority", "Steps", "Tags"])
        for tc in self.test_cases:
            steps_text = " | ".join(f"{s.step_number}. {s.action}" for s in tc.steps)
            writer.writerow([tc.id, tc.title, tc.type.value, tc.priority.value, steps_text, ",".join(tc.tags)])
        return buf.getvalue()


@dataclass
class CoverageGap:
    """A gap in test coverage."""
    area: str
    description: str
    severity: str = "medium"
    suggested_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"area": self.area, "description": self.description, "severity": self.severity, "suggested_tests": self.suggested_tests}


@dataclass
class QAResult:
    """Complete result from QA agent analysis."""
    suite: TestSuite = field(default_factory=TestSuite)
    coverage_gaps: list[CoverageGap] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)
    risk_areas: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    raw_analysis: str = ""

    def to_dict(self) -> dict:
        return {
            "suite": self.suite.to_dict(),
            "coverage_gaps": [g.to_dict() for g in self.coverage_gaps],
            "edge_cases": self.edge_cases,
            "risk_areas": self.risk_areas,
            "recommendations": self.recommendations,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  QA Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QAAgent:
    """AI-powered QA testing agent.

    Generates test cases, discovers edge cases, analyzes code, and produces
    comprehensive test suites from PRD/BRD documents or natural language descriptions.

    Usage:
        agent = QAAgent.create()
        result = await agent.generate_tests("User login with email and password")
        result = await agent.generate_from_prd("path/to/prd.pdf")
        result = await agent.analyze_github("https://github.com/user/repo")
        result = await agent.generate_edge_cases("Payment checkout flow")
    """

    SYSTEM_PROMPT = """You are an expert QA engineer and test architect with 15 years of experience.

Your capabilities:
1. Parse PRD/BRD documents to extract testable requirements
2. Generate comprehensive test cases (functional, edge, regression, API, UI, security, performance)
3. Discover edge cases that most testers miss
4. Analyze code to identify untested paths
5. Create test data and fixtures
6. Produce test execution plans
7. Identify coverage gaps and risk areas

When generating test cases, always include:
- Clear, numbered steps with actions and expected results
- Specific test data (not generic placeholders)
- Preconditions and postconditions
- Priority and type classification
- Edge cases and boundary conditions

Output format: Always return structured JSON when generating test cases.
Be thorough, precise, and think like a senior QA engineer who needs to ensure zero production bugs."""

    def __init__(self, agent=None, llm_provider: str = "openai", llm_model: str = "gpt-4o"):
        self._agent = agent
        self._llm_provider = llm_provider
        self._llm_model = llm_model

    @classmethod
    def create(cls, llm_provider: str = "openai", llm_model: str = "gpt-4o", tools: list | None = None):
        """Create a QA Agent with full capabilities."""
        from duxx_ai.core.agent import Agent, AgentConfig
        from duxx_ai.core.llm import LLMConfig

        all_tools = []
        # Add file reading tools if available
        try:
            from duxx_ai.tools.builtin import get_builtin_tools
            all_tools.extend(get_builtin_tools(["read_file", "list_files", "web_request"]))
        except:
            pass
        if tools:
            all_tools.extend(tools)

        agent = Agent(
            config=AgentConfig(
                name="qa-agent",
                system_prompt=cls.SYSTEM_PROMPT,
                llm=LLMConfig(provider=llm_provider, model=llm_model),
                max_iterations=15,
            ),
            tools=all_tools,
        )
        return cls(agent=agent, llm_provider=llm_provider, llm_model=llm_model)

    async def _ask(self, prompt: str) -> str:
        """Internal: ask the LLM agent."""
        if self._agent:
            self._agent.reset()
            return await self._agent.run(prompt)
        # Fallback: direct LLM call
        from duxx_ai.core.llm import LLMConfig, create_provider
        from duxx_ai.core.message import Conversation, Message, Role
        provider = create_provider(LLMConfig(provider=self._llm_provider, model=self._llm_model))
        conv = Conversation()
        conv.add(Message(role=Role.SYSTEM, content=self.SYSTEM_PROMPT))
        conv.add(Message(role=Role.USER, content=prompt))
        resp = await provider.chat(conv, max_tokens=4096)
        return resp.content

    def _parse_test_cases(self, raw: str) -> list[TestCase]:
        """Parse LLM output into structured test cases."""
        cases = []
        # Try JSON parsing first
        try:
            json_match = re.search(r'\[[\s\S]*\]', raw)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    tc = TestCase(
                        title=item.get("title", ""),
                        description=item.get("description", ""),
                        type=TestType(item.get("type", "functional")),
                        priority=TestPriority(item.get("priority", "medium")),
                        preconditions=item.get("preconditions", []),
                        tags=item.get("tags", []),
                    )
                    for i, step in enumerate(item.get("steps", []), 1):
                        if isinstance(step, dict):
                            tc.steps.append(TestStep(
                                step_number=step.get("step", i),
                                action=step.get("action", ""),
                                expected_result=step.get("expected", step.get("expected_result", "")),
                                test_data=step.get("data", step.get("test_data", "")),
                            ))
                        elif isinstance(step, str):
                            tc.steps.append(TestStep(step_number=i, action=step, expected_result=""))
                    cases.append(tc)
                return cases
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: parse numbered test cases from text
        tc_blocks = re.split(r'\n(?=(?:Test Case|TC|#)\s*\d)', raw)
        for block in tc_blocks:
            if not block.strip():
                continue
            title_match = re.search(r'(?:Test Case|TC|#)\s*\d+[:.]\s*(.+)', block)
            if title_match:
                tc = TestCase(title=title_match.group(1).strip())
                step_matches = re.findall(r'(?:Step\s*)?(\d+)[.:)]\s*(.+?)(?:\n|$)', block)
                for i, (num, text) in enumerate(step_matches):
                    tc.steps.append(TestStep(step_number=int(num), action=text.strip(), expected_result=""))
                if tc.steps:
                    cases.append(tc)

        return cases

    # ── Main Methods ──

    async def generate_tests(self, description: str, test_types: list[str] | None = None, count: int = 10) -> QAResult:
        """Generate test cases from a natural language description.

        Args:
            description: Feature or requirement description
            test_types: Types of tests to generate (functional, edge_case, regression, etc.)
            count: Approximate number of test cases to generate

        Returns:
            QAResult with test suite, edge cases, and recommendations.
        """
        types_str = ", ".join(test_types) if test_types else "functional, edge_case, regression, API, UI, security"

        prompt = f"""Generate {count} comprehensive test cases for the following feature/requirement:

DESCRIPTION: {description}

TEST TYPES TO INCLUDE: {types_str}

Return a JSON array where each test case has:
{{
    "title": "Clear test case title",
    "description": "What this test verifies",
    "type": "functional|edge_case|regression|api|ui|security|performance",
    "priority": "critical|high|medium|low",
    "preconditions": ["List of preconditions"],
    "steps": [
        {{"step": 1, "action": "What to do", "expected": "What should happen", "data": "Test data if any"}}
    ],
    "tags": ["relevant", "tags"]
}}

Include:
- Happy path scenarios
- Edge cases and boundary conditions
- Error handling scenarios
- Security considerations
- Performance edge cases
- Data validation tests

Return ONLY the JSON array."""

        raw = await self._ask(prompt)
        cases = self._parse_test_cases(raw)

        # Generate edge cases
        edge_prompt = f"""For this feature: {description}

List 10 edge cases that most QA testers would miss. Think about:
- Boundary values, null/empty inputs, special characters
- Concurrent operations, race conditions
- Network failures, timeouts, partial data
- Localization, timezone, currency issues
- Browser/device compatibility
- Accessibility concerns
- Data migration / backward compatibility

Return a JSON array of strings: ["edge case 1", "edge case 2", ...]"""

        edge_raw = await self._ask(edge_prompt)
        edge_cases = []
        try:
            edge_match = re.search(r'\[[\s\S]*\]', edge_raw)
            if edge_match:
                edge_cases = json.loads(edge_match.group())
        except:
            edge_cases = [line.strip().lstrip("- ") for line in edge_raw.split("\n") if line.strip() and not line.startswith("{")]

        suite = TestSuite(
            name=f"Test Suite: {description[:60]}",
            description=description,
            test_cases=cases,
        )

        return QAResult(
            suite=suite,
            edge_cases=edge_cases[:15],
            recommendations=[
                f"Generated {len(cases)} test cases covering {len(suite.by_type)} test types",
                f"Priority breakdown: {suite.by_priority}",
                "Review edge cases for additional coverage",
            ],
            raw_analysis=raw,
        )

    async def generate_from_prd(self, prd_text: str) -> QAResult:
        """Generate test cases from a PRD/BRD document text.

        Args:
            prd_text: Full text of the PRD or BRD document

        Returns:
            QAResult with requirements, test cases, coverage gaps.
        """
        # Phase 1: Extract requirements
        req_prompt = f"""Analyze this PRD/BRD document and extract ALL testable requirements.

DOCUMENT:
{prd_text[:8000]}

Return a JSON array of requirements:
[
    {{
        "id": "REQ-001",
        "text": "The requirement description",
        "category": "authentication|payment|ui|api|data|security|performance",
        "priority": "critical|high|medium|low",
        "acceptance_criteria": ["AC1: ...", "AC2: ..."]
    }}
]

Extract every testable requirement, including implicit ones. Return ONLY JSON."""

        req_raw = await self._ask(req_prompt)
        requirements = []
        try:
            req_match = re.search(r'\[[\s\S]*\]', req_raw)
            if req_match:
                req_data = json.loads(req_match.group())
                for r in req_data:
                    requirements.append(Requirement(
                        id=r.get("id", f"REQ-{len(requirements)+1:03d}"),
                        text=r.get("text", ""),
                        category=r.get("category", ""),
                        priority=r.get("priority", "medium"),
                        acceptance_criteria=r.get("acceptance_criteria", []),
                    ))
        except:
            pass

        # Phase 2: Generate test cases for each requirement
        all_cases = []
        for req in requirements[:20]:  # Limit to avoid token overflow
            tc_prompt = f"""Generate 3-5 test cases for this requirement:

Requirement ID: {req.id}
Requirement: {req.text}
Category: {req.category}
Acceptance Criteria: {json.dumps(req.acceptance_criteria)}

Return JSON array of test cases (same format as before). Include edge cases."""

            tc_raw = await self._ask(tc_prompt)
            cases = self._parse_test_cases(tc_raw)
            for tc in cases:
                tc.requirement_id = req.id
            all_cases.extend(cases)
            req.test_cases = cases

        # Phase 3: Coverage analysis
        gap_prompt = f"""Analyze test coverage for these requirements and identify gaps:

Requirements: {json.dumps([r.to_dict() for r in requirements[:15]])}
Test cases generated: {len(all_cases)}

Identify:
1. Requirements with insufficient test coverage
2. Missing test types (security, performance, accessibility)
3. Integration test gaps
4. Risk areas

Return JSON:
{{
    "coverage_gaps": [{{"area": "...", "description": "...", "severity": "high|medium|low", "suggested_tests": ["..."]}}],
    "risk_areas": ["..."],
    "recommendations": ["..."]
}}"""

        gap_raw = await self._ask(gap_prompt)
        coverage_gaps = []
        risk_areas = []
        recommendations = []
        try:
            gap_match = re.search(r'\{[\s\S]*\}', gap_raw)
            if gap_match:
                gap_data = json.loads(gap_match.group())
                for g in gap_data.get("coverage_gaps", []):
                    coverage_gaps.append(CoverageGap(
                        area=g.get("area", ""),
                        description=g.get("description", ""),
                        severity=g.get("severity", "medium"),
                        suggested_tests=g.get("suggested_tests", []),
                    ))
                risk_areas = gap_data.get("risk_areas", [])
                recommendations = gap_data.get("recommendations", [])
        except:
            pass

        suite = TestSuite(
            name=f"PRD Test Suite",
            description=f"Generated from PRD with {len(requirements)} requirements",
            test_cases=all_cases,
            requirements=requirements,
            coverage_summary={
                "requirements": len(requirements),
                "test_cases": len(all_cases),
                "gaps": len(coverage_gaps),
            },
        )

        return QAResult(
            suite=suite,
            coverage_gaps=coverage_gaps,
            risk_areas=risk_areas,
            recommendations=recommendations,
        )

    async def generate_edge_cases(self, feature: str, count: int = 20) -> QAResult:
        """Generate edge cases for a feature.

        Args:
            feature: Description of the feature to find edge cases for
            count: Number of edge cases to generate

        Returns:
            QAResult focused on edge cases.
        """
        prompt = f"""You are an expert QA engineer specializing in edge case discovery.

For this feature: {feature}

Generate {count} edge cases that would catch production bugs. Think about:

1. BOUNDARY VALUES: Min, max, zero, negative, overflow, underflow
2. INPUT VALIDATION: Null, empty, whitespace, special chars, unicode, emoji, SQL injection, XSS
3. STATE: Concurrent access, race conditions, stale data, cache invalidation
4. NETWORK: Timeout, partial response, disconnection, retry storms
5. DATA: Large payloads, malformed JSON, missing fields, extra fields, type coercion
6. TIME: Timezones, daylight saving, leap years, epoch boundaries, future dates
7. PERMISSIONS: Unauthorized access, role escalation, expired tokens, CORS
8. RESOURCES: Memory limits, disk full, CPU throttling, connection pool exhaustion
9. INTEGRATION: API version mismatch, schema changes, third-party outage
10. USER BEHAVIOR: Double-click, back button, tab switching, copy-paste, autofill

For each edge case, generate a complete test case.

Return JSON array:
[{{"title": "...", "description": "...", "type": "edge_case", "priority": "...", "steps": [{{"step": 1, "action": "...", "expected": "..."}}], "tags": ["..."]}}]"""

        raw = await self._ask(prompt)
        cases = self._parse_test_cases(raw)
        for tc in cases:
            tc.type = TestType.EDGE_CASE

        suite = TestSuite(
            name=f"Edge Cases: {feature[:50]}",
            description=f"Edge case analysis for: {feature}",
            test_cases=cases,
        )

        return QAResult(
            suite=suite,
            edge_cases=[tc.title for tc in cases],
        )

    async def analyze_github(self, repo_url: str, focus: str = "") -> QAResult:
        """Analyze a GitHub repository and generate test cases.

        Args:
            repo_url: GitHub repository URL
            focus: Optional focus area (e.g., "authentication", "payment")

        Returns:
            QAResult with code-aware test cases.
        """
        # Extract owner/repo from URL
        match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            return QAResult(recommendations=["Invalid GitHub URL"])

        owner, repo = match.group(1), match.group(2)

        prompt = f"""Analyze the GitHub repository: {owner}/{repo}

{f'Focus on: {focus}' if focus else ''}

Based on common patterns for this type of repository:
1. Identify the main features and user flows
2. Generate comprehensive test cases
3. Identify potential edge cases and security concerns
4. Suggest integration and API tests

Generate 15 test cases covering:
- Core functionality (happy path)
- Error handling
- Edge cases
- Security (auth, input validation)
- API endpoints
- Performance concerns

Return JSON array of test cases."""

        raw = await self._ask(prompt)
        cases = self._parse_test_cases(raw)

        suite = TestSuite(
            name=f"Tests: {owner}/{repo}",
            description=f"Auto-generated tests for {repo_url}",
            test_cases=cases,
            metadata={"repo": repo_url, "owner": owner, "repo_name": repo},
        )

        return QAResult(suite=suite, raw_analysis=raw)

    async def analyze_test_results(self, results: list[dict]) -> dict[str, Any]:
        """Analyze test execution results and suggest fixes.

        Args:
            results: List of {"test": "name", "status": "passed|failed", "error": "..."}

        Returns:
            Analysis with failure patterns, root causes, and fix suggestions.
        """
        prompt = f"""Analyze these test results and provide insights:

{json.dumps(results[:50], indent=2)}

Provide:
1. Summary of results (pass/fail counts)
2. Pattern analysis (common failure patterns)
3. Root cause analysis for failures
4. Prioritized fix suggestions
5. Risk assessment

Return JSON:
{{
    "summary": {{"total": N, "passed": N, "failed": N, "pass_rate": "X%"}},
    "failure_patterns": ["pattern1", "pattern2"],
    "root_causes": [{{"test": "name", "cause": "likely root cause", "fix": "suggested fix"}}],
    "risk_assessment": "overall risk level and explanation",
    "recommendations": ["prioritized action items"]
}}"""

        raw = await self._ask(prompt)
        try:
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {"raw": raw}

    async def generate_test_data(self, schema: str, count: int = 10) -> list[dict]:
        """Generate realistic test data based on a schema.

        Args:
            schema: Description or JSON schema of the data needed
            count: Number of records to generate

        Returns:
            List of test data records.
        """
        prompt = f"""Generate {count} realistic test data records for:

Schema: {schema}

Include:
- Normal/valid data (70%)
- Edge cases (20%): boundary values, special characters, very long/short values
- Invalid data (10%): for negative testing

Return a JSON array of objects matching the schema."""

        raw = await self._ask(prompt)
        try:
            match = re.search(r'\[[\s\S]*\]', raw)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []

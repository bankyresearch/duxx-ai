"""Audit logging for enterprise compliance and traceability."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    event_type: str
    agent_name: str = ""
    user_id: str = ""
    action: str = ""
    input_summary: str = ""
    output_summary: str = ""
    tool_name: str | None = None
    guardrail_result: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high, critical


class AuditLog:
    """Append-only audit log for compliance and debugging."""

    def __init__(self, storage_path: str | None = None) -> None:
        self.entries: list[AuditEntry] = []
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        self.entries.append(entry)
        if self.storage_path:
            with open(self.storage_path, "a") as f:
                f.write(entry.model_dump_json() + "\n")

    def log_agent_run(
        self,
        agent_name: str,
        user_input: str,
        output: str,
        user_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.log(
            AuditEntry(
                event_type="agent_run",
                agent_name=agent_name,
                user_id=user_id,
                action="run",
                input_summary=user_input[:200],
                output_summary=output[:200],
                metadata=metadata or {},
            )
        )

    def log_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        risk_level: str = "low",
    ) -> None:
        self.log(
            AuditEntry(
                event_type="tool_call",
                agent_name=agent_name,
                tool_name=tool_name,
                action="execute_tool",
                input_summary=json.dumps(arguments)[:200],
                output_summary=str(result)[:200] if result else "",
                risk_level=risk_level,
                metadata={"error": error} if error else {},
            )
        )

    def log_guardrail_trigger(
        self,
        agent_name: str,
        guardrail_name: str,
        direction: str,
        blocked: bool,
        reason: str = "",
    ) -> None:
        self.log(
            AuditEntry(
                event_type="guardrail_trigger",
                agent_name=agent_name,
                action=f"guardrail_{direction}",
                guardrail_result="blocked" if blocked else "passed",
                risk_level="high" if blocked else "low",
                metadata={"guardrail": guardrail_name, "reason": reason},
            )
        )

    def query(
        self,
        event_type: str | None = None,
        agent_name: str | None = None,
        risk_level: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        results = self.entries
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if agent_name:
            results = [e for e in results if e.agent_name == agent_name]
        if risk_level:
            results = [e for e in results if e.risk_level == risk_level]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results[-limit:]

    def get_risk_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for e in self.entries:
            summary[e.risk_level] = summary.get(e.risk_level, 0) + 1
        return summary

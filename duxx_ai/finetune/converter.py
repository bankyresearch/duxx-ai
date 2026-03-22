"""Format converter — convert raw data from any format to model-ready training JSONL.

Supports auto-detection and conversion of CSV, Excel, Parquet, JSON, text, SQL results,
and HuggingFace datasets into chat, instruction, or text training formats.

Includes ColumnMapper for interactive column mapping with heuristic suggestions.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Common column name patterns for auto-mapping
_PATTERNS = {
    "user": re.compile(r"(?i)(question|query|prompt|user|input|instruction|request|ask|human|problem|source)"),
    "assistant": re.compile(r"(?i)(answer|response|output|assistant|reply|completion|target|solution|result|bot)"),
    "system": re.compile(r"(?i)(system|context|system_prompt|system_message|preamble)"),
    "input": re.compile(r"(?i)(input|context|additional|extra|supporting|reference)"),
    "text": re.compile(r"(?i)(text|content|body|passage|document|article|paragraph|description)"),
    "messages": re.compile(r"(?i)(messages|conversation|chat|dialogue|turns)"),
}


class ColumnMapper:
    """Map raw data columns to training format fields using heuristic detection."""

    @staticmethod
    def suggest_mapping(columns: list[str]) -> dict[str, str]:
        """Suggest column → field mapping based on column names.

        Returns dict like {"question": "user", "answer": "assistant"} mapping
        raw column names to training roles.
        """
        mapping: dict[str, str] = {}
        used_roles: set[str] = set()

        # First pass: exact/strong matches
        for col in columns:
            for role, pattern in _PATTERNS.items():
                if role not in used_roles and pattern.search(col):
                    mapping[col] = role
                    used_roles.add(role)
                    break

        # If we found both user and assistant columns, we have a good instruction mapping
        # If we only found text, it's a text format
        return mapping

    @staticmethod
    def detect_format(columns: list[str], sample_data: list[dict[str, Any]] | None = None) -> str:
        """Detect the most likely training format based on columns and sample data.

        Returns: 'chat', 'instruction', 'text', or 'raw'
        """
        col_set = set(c.lower() for c in columns)

        # Check for chat format
        if "messages" in col_set or "conversation" in col_set or "conversations" in col_set:
            return "chat"

        # Check for instruction format
        mapping = ColumnMapper.suggest_mapping(columns)
        roles = set(mapping.values())
        if "user" in roles and "assistant" in roles:
            return "instruction"

        # Check if data has 'messages' key in samples
        if sample_data:
            first = sample_data[0]
            if "messages" in first:
                return "chat"
            if "instruction" in first or "question" in first:
                return "instruction"
            if "text" in first:
                return "text"

        # Single text column
        if len(columns) == 1:
            return "text"

        return "raw"

    @staticmethod
    def apply_mapping(
        data: list[dict[str, Any]],
        mapping: dict[str, str],
        target_format: str = "chat",
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> list[dict[str, Any]]:
        """Apply column mapping to convert raw data to training format."""
        # Reverse mapping: role → column_name
        role_to_col: dict[str, str] = {v: k for k, v in mapping.items()}

        if target_format == "chat":
            return _to_chat(data, role_to_col, system_prompt)
        elif target_format == "instruction":
            return _to_instruction(data, role_to_col)
        elif target_format == "text":
            return _to_text(data, role_to_col)
        else:
            return data


class FormatConverter:
    """Convert raw data from any format to model-ready training JSONL."""

    @staticmethod
    def auto_convert(
        data: list[dict[str, Any]],
        target_format: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        column_mapping: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Auto-detect format and convert to training-ready samples.

        Args:
            data: Raw data as list of dicts
            target_format: Force output format (chat/instruction/text). Auto-detect if None.
            system_prompt: System prompt for chat format
            column_mapping: Manual column → role mapping. Auto-detect if None.

        Returns:
            List of training-ready samples in the target format
        """
        if not data:
            return []

        columns = list(data[0].keys())

        # Check if already in training format
        first = data[0]
        if "messages" in first and isinstance(first["messages"], list):
            if target_format and target_format != "chat":
                return _convert_between_formats(data, "chat", target_format)
            return data
        if "instruction" in first and "output" in first:
            if target_format and target_format != "instruction":
                return _convert_between_formats(data, "instruction", target_format)
            return data
        if "text" in first and len(first) == 1:
            if target_format and target_format != "text":
                return _convert_between_formats(data, "text", target_format)
            return data

        # Auto-detect or use provided mapping
        if column_mapping is None:
            column_mapping = ColumnMapper.suggest_mapping(columns)

        if target_format is None:
            target_format = ColumnMapper.detect_format(columns, data)

        if target_format == "raw":
            # Fallback: concatenate all values as text
            target_format = "text"
            if not column_mapping:
                column_mapping = {columns[0]: "text"}

        return ColumnMapper.apply_mapping(data, column_mapping, target_format, system_prompt)

    @staticmethod
    def convert_hf_dataset(
        data: list[dict[str, Any]],
        dataset_name: str = "",
        target_format: str = "chat",
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> list[dict[str, Any]]:
        """Convert a HuggingFace dataset to training format with dataset-specific handling."""
        if not data:
            return []

        first = data[0]

        # Already in messages format (e.g., OpenAssistant, UltraChat)
        if "messages" in first or "conversations" in first:
            key = "messages" if "messages" in first else "conversations"
            result = []
            for row in data:
                msgs = row.get(key, [])
                if isinstance(msgs, list) and len(msgs) >= 2:
                    result.append({"messages": msgs})
            return result

        # Alpaca-style: instruction, input, output
        if "instruction" in first and "output" in first:
            if target_format == "chat":
                return _to_chat(data, {"user": "instruction", "assistant": "output", "input": "input"}, system_prompt)
            return [{"instruction": r["instruction"], "input": r.get("input", ""), "output": r["output"]} for r in data]

        # Q&A style: question, answer
        if "question" in first and "answer" in first:
            return _to_chat(data, {"user": "question", "assistant": "answer"}, system_prompt)

        # Text-only
        if "text" in first:
            return [{"text": r["text"]} for r in data if r.get("text")]

        # Fallback: auto-convert
        return FormatConverter.auto_convert(data, target_format, system_prompt)

    @staticmethod
    def get_columns(data: list[dict[str, Any]]) -> list[str]:
        """Get column names from data."""
        if not data:
            return []
        return list(data[0].keys())

    @staticmethod
    def preview_conversion(
        data: list[dict[str, Any]],
        mapping: dict[str, str],
        target_format: str = "chat",
        n: int = 3,
    ) -> list[dict[str, Any]]:
        """Preview the first N converted samples."""
        converted = ColumnMapper.apply_mapping(data[:n], mapping, target_format)
        return converted


# ── Internal conversion helpers ──

def _to_chat(
    data: list[dict[str, Any]],
    role_to_col: dict[str, str],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Convert to chat format with messages array."""
    results = []
    user_col = role_to_col.get("user", "")
    asst_col = role_to_col.get("assistant", "")
    input_col = role_to_col.get("input", "")

    for row in data:
        user_text = str(row.get(user_col, "")).strip()
        asst_text = str(row.get(asst_col, "")).strip()
        if not user_text or not asst_text:
            continue

        # Add input context if available
        if input_col and row.get(input_col):
            user_text = f"{user_text}\n\nContext: {row[input_col]}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ]
        results.append({"messages": messages})

    return results


def _to_instruction(
    data: list[dict[str, Any]],
    role_to_col: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert to instruction format."""
    results = []
    user_col = role_to_col.get("user", "")
    asst_col = role_to_col.get("assistant", "")
    input_col = role_to_col.get("input", "")

    for row in data:
        instruction = str(row.get(user_col, "")).strip()
        output = str(row.get(asst_col, "")).strip()
        if not instruction:
            continue
        entry: dict[str, Any] = {"instruction": instruction, "output": output}
        if input_col and row.get(input_col):
            entry["input"] = str(row[input_col])
        results.append(entry)

    return results


def _to_text(
    data: list[dict[str, Any]],
    role_to_col: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert to text format."""
    text_col = role_to_col.get("text", "")
    results = []
    for row in data:
        if text_col and text_col in row:
            text = str(row[text_col]).strip()
        else:
            # Concatenate all values
            text = " ".join(str(v) for v in row.values() if v).strip()
        if text:
            results.append({"text": text})
    return results


def _convert_between_formats(
    data: list[dict[str, Any]],
    from_format: str,
    to_format: str,
) -> list[dict[str, Any]]:
    """Convert between training formats (chat ↔ instruction ↔ text)."""
    results = []
    for sample in data:
        if from_format == "chat" and "messages" in sample:
            msgs = sample["messages"]
            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            if to_format == "instruction":
                results.append({"instruction": user_msg, "output": asst_msg})
            elif to_format == "text":
                results.append({"text": f"User: {user_msg}\nAssistant: {asst_msg}"})

        elif from_format == "instruction":
            inst = sample.get("instruction", "")
            out = sample.get("output", "")
            if to_format == "chat":
                results.append({"messages": [
                    {"role": "user", "content": inst},
                    {"role": "assistant", "content": out},
                ]})
            elif to_format == "text":
                results.append({"text": f"Instruction: {inst}\nResponse: {out}"})

        elif from_format == "text":
            text = sample.get("text", "")
            if to_format == "chat":
                results.append({"messages": [
                    {"role": "user", "content": "Continue the following:"},
                    {"role": "assistant", "content": text},
                ]})
            elif to_format == "instruction":
                results.append({"instruction": "Summarize or continue:", "output": text})

    return results

"""Output parsers — extract structured data from LLM responses.

Provides JSON, Pydantic, Markdown, and Regex parsers with automatic
retry and error correction support.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, Type

from pydantic import BaseModel

T = TypeVar("T")


class ParseError(Exception):
    """Raised when an output parser fails to extract structured data."""
    pass


class OutputParser(ABC, Generic[T]):
    """Base class for all output parsers."""

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse LLM output text into structured data."""
        ...

    def get_format_instructions(self) -> str:
        """Return instructions to include in the prompt for proper formatting."""
        return ""


class JSONOutputParser(OutputParser[dict[str, Any]]):
    """Extract JSON from LLM output, handling markdown code fences."""

    def parse(self, text: str) -> dict[str, Any]:
        # Try direct parse first
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fences
        patterns = [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if "```" in pattern else match.group(0))
                except (json.JSONDecodeError, IndexError):
                    continue

        raise ParseError(f"Could not extract JSON from text: {text[:200]}...")

    def get_format_instructions(self) -> str:
        return "Respond with a JSON object. Use ```json code fences."


class PydanticOutputParser(OutputParser[T]):
    """Parse LLM output into a Pydantic model with schema validation."""

    def __init__(self, model_class: Type[BaseModel]) -> None:
        self.model_class = model_class
        self._json_parser = JSONOutputParser()

    def parse(self, text: str) -> Any:
        data = self._json_parser.parse(text)
        try:
            return self.model_class.model_validate(data)
        except Exception as e:
            raise ParseError(f"Pydantic validation failed: {e}")

    def get_format_instructions(self) -> str:
        schema = self.model_class.model_json_schema()
        fields = []
        for name, prop in schema.get("properties", {}).items():
            ptype = prop.get("type", "string")
            desc = prop.get("description", "")
            fields.append(f'  "{name}": <{ptype}>{" // " + desc if desc else ""}')
        return (
            f"Respond with a JSON object matching this schema:\n"
            f"{{\n" + ",\n".join(fields) + "\n}"
        )


class MarkdownOutputParser(OutputParser[dict[str, str]]):
    """Extract structured sections from markdown-formatted output."""

    def __init__(self, sections: list[str] | None = None) -> None:
        self.sections = sections

    def parse(self, text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        # Split by ## or # headers
        parts = re.split(r"^#{1,3}\s+(.+)$", text, flags=re.MULTILINE)

        # parts[0] = preamble, then alternating header/content
        if len(parts) > 1:
            for i in range(1, len(parts) - 1, 2):
                header = parts[i].strip()
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                result[header] = content
        else:
            result["content"] = text.strip()

        if self.sections:
            filtered = {}
            for s in self.sections:
                for k, v in result.items():
                    if s.lower() in k.lower():
                        filtered[s] = v
                        break
            return filtered
        return result

    def get_format_instructions(self) -> str:
        if self.sections:
            headers = "\n".join(f"## {s}" for s in self.sections)
            return f"Format your response with these markdown sections:\n{headers}"
        return "Format your response with markdown headers (## Section Name)."


class RegexOutputParser(OutputParser[dict[str, str]]):
    """Extract named groups from LLM output using regex patterns."""

    def __init__(self, pattern: str) -> None:
        self.pattern = re.compile(pattern, re.DOTALL)

    def parse(self, text: str) -> dict[str, str]:
        match = self.pattern.search(text)
        if not match:
            raise ParseError(f"Pattern did not match: {self.pattern.pattern}")
        return match.groupdict()


class ListOutputParser(OutputParser[list[str]]):
    """Extract a list of items from LLM output (numbered or bulleted)."""

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        items = []
        for line in lines:
            # Remove numbering, bullets, dashes
            cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*+]\s*)", "", line).strip()
            if cleaned:
                items.append(cleaned)
        return items

    def get_format_instructions(self) -> str:
        return "Respond with a numbered list:\n1. First item\n2. Second item"


class RetryParser(OutputParser[T]):
    """Wraps another parser and retries on failure with error feedback."""

    def __init__(self, parser: OutputParser[T], max_retries: int = 2) -> None:
        self.parser = parser
        self.max_retries = max_retries
        self._last_errors: list[str] = []

    def parse(self, text: str) -> T:
        try:
            return self.parser.parse(text)
        except ParseError as e:
            self._last_errors.append(str(e))
            raise

    @property
    def last_errors(self) -> list[str]:
        return self._last_errors

    def get_format_instructions(self) -> str:
        base = self.parser.get_format_instructions()
        if self._last_errors:
            base += f"\n\nPrevious attempt failed: {self._last_errors[-1]}\nPlease fix the formatting."
        return base

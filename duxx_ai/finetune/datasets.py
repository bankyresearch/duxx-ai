"""Dataset manager — load, validate, split, preview, and export datasets for fine-tuning.

Supports JSONL format with chat (messages), instruction (instruction/output),
and raw text (text) formats. Provides train/validation/test splitting,
statistics, and format validation.
"""

from __future__ import annotations

import json
import logging
import math
import random
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DatasetStats(BaseModel):
    """Statistics about a dataset."""
    total_samples: int = 0
    format_counts: dict[str, int] = Field(default_factory=dict)  # chat, instruction, text
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    total_tokens: int = 0
    avg_messages_per_sample: float = 0.0
    token_distribution: dict[str, int] = Field(default_factory=dict)  # bucket -> count


class ValidationResult(BaseModel):
    """Result of dataset validation."""
    valid: bool = True
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    detected_format: str = ""


class SplitResult(BaseModel):
    """Result of dataset splitting."""
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""


class DatasetEntry(BaseModel):
    """A single dataset entry."""
    data: dict[str, Any] = Field(default_factory=dict)
    format: str = ""  # chat, instruction, text
    token_count: int = 0
    line_number: int = 0


class DatasetInfo(BaseModel):
    """Metadata about a managed dataset."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    source_path: str = ""
    format: str = ""
    total_samples: int = 0
    status: str = "loaded"  # loaded, validated, split, error
    created_at: float = 0.0


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def _detect_format(sample: dict[str, Any]) -> str:
    """Detect the format of a single sample."""
    if "messages" in sample and isinstance(sample["messages"], list):
        return "chat"
    if "instruction" in sample:
        return "instruction"
    if "text" in sample:
        return "text"
    return "unknown"


def _extract_text(sample: dict[str, Any]) -> str:
    """Extract the full text content from a sample for token counting."""
    fmt = _detect_format(sample)
    if fmt == "chat":
        return " ".join(m.get("content", "") for m in sample.get("messages", []))
    elif fmt == "instruction":
        return sample.get("instruction", "") + " " + sample.get("output", "")
    elif fmt == "text":
        return sample.get("text", "")
    return json.dumps(sample)


class DatasetManager:
    """Load, validate, split, preview, and export fine-tuning datasets."""

    def __init__(self) -> None:
        self.entries: list[DatasetEntry] = []
        self.info = DatasetInfo()

    def load_jsonl(self, path: str, name: str = "") -> int:
        """Load a JSONL file and return the number of samples loaded."""
        self.entries = []
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(file_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    fmt = _detect_format(data)
                    text = _extract_text(data)
                    tokens = _estimate_tokens(text)
                    self.entries.append(DatasetEntry(
                        data=data, format=fmt, token_count=tokens, line_number=i + 1
                    ))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON at line {i + 1}")

        self.info = DatasetInfo(
            name=name or file_path.stem,
            source_path=str(file_path),
            format=self.entries[0].format if self.entries else "unknown",
            total_samples=len(self.entries),
            status="loaded",
        )
        import time
        self.info.created_at = time.time()
        return len(self.entries)

    def load_from_connector(self, connector: Any, name: str = "", max_samples: int | None = None) -> int:
        """Load data from any DataConnector (S3, GCS, Azure, HuggingFace, DB, etc.)."""
        raw_data = connector.load(max_samples=max_samples)
        src_type = getattr(connector, "source_type", "connector")
        return self.load_from_list(raw_data, name=name or f"{src_type}-dataset")

    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        max_samples: int | None = None,
        auto_convert: bool = True,
    ) -> int:
        """Load a dataset from HuggingFace Hub and optionally auto-convert to training format."""
        from duxx_ai.finetune.connectors import HuggingFaceConnector
        connector = HuggingFaceConnector(dataset_name, split=split, subset=subset)
        raw_data = connector.load(max_samples=max_samples)

        if auto_convert:
            from duxx_ai.finetune.converter import FormatConverter
            raw_data = FormatConverter.convert_hf_dataset(raw_data, dataset_name=dataset_name)

        return self.load_from_list(raw_data, name=dataset_name.split("/")[-1])

    def load_from_list(self, samples: list[dict[str, Any]], name: str = "in-memory") -> int:
        """Load from a list of dicts (for programmatic use)."""
        self.entries = []
        for i, data in enumerate(samples):
            fmt = _detect_format(data)
            text = _extract_text(data)
            tokens = _estimate_tokens(text)
            self.entries.append(DatasetEntry(data=data, format=fmt, token_count=tokens, line_number=i + 1))

        self.info = DatasetInfo(
            name=name, format=self.entries[0].format if self.entries else "unknown",
            total_samples=len(self.entries), status="loaded",
        )
        return len(self.entries)

    def preview(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the first N samples for preview."""
        return [
            {
                "line": e.line_number,
                "format": e.format,
                "tokens": e.token_count,
                "data": e.data,
            }
            for e in self.entries[:n]
        ]

    def validate(self) -> ValidationResult:
        """Validate dataset format and quality."""
        result = ValidationResult(total_samples=len(self.entries))

        if not self.entries:
            result.valid = False
            result.errors.append("Dataset is empty")
            return result

        format_counts: dict[str, int] = {}
        for entry in self.entries:
            format_counts[entry.format] = format_counts.get(entry.format, 0) + 1

            if entry.format == "chat":
                msgs = entry.data.get("messages", [])
                if len(msgs) < 2:
                    result.invalid_samples += 1
                    result.errors.append(f"Line {entry.line_number}: chat needs at least 2 messages")
                    continue
                roles = [m.get("role") for m in msgs]
                if "assistant" not in roles:
                    result.invalid_samples += 1
                    result.errors.append(f"Line {entry.line_number}: missing assistant message")
                    continue
            elif entry.format == "instruction":
                if not entry.data.get("instruction"):
                    result.invalid_samples += 1
                    result.errors.append(f"Line {entry.line_number}: empty instruction")
                    continue
                if not entry.data.get("output"):
                    result.warnings.append(f"Line {entry.line_number}: empty output")
            elif entry.format == "text":
                if len(entry.data.get("text", "")) < 10:
                    result.warnings.append(f"Line {entry.line_number}: very short text (<10 chars)")
            elif entry.format == "unknown":
                result.invalid_samples += 1
                result.errors.append(f"Line {entry.line_number}: unrecognized format")
                continue

            result.valid_samples += 1

        # Determine dominant format
        if format_counts:
            result.detected_format = max(format_counts, key=format_counts.get)  # type: ignore

        # Limit error list
        if len(result.errors) > 20:
            total_errors = len(result.errors)
            result.errors = result.errors[:20]
            result.errors.append(f"... and {total_errors - 20} more errors")

        result.valid = result.invalid_samples == 0
        self.info.status = "validated" if result.valid else "error"
        self.info.format = result.detected_format
        return result

    def stats(self) -> DatasetStats:
        """Compute dataset statistics."""
        if not self.entries:
            return DatasetStats()

        tokens = [e.token_count for e in self.entries]
        format_counts: dict[str, int] = {}
        total_messages = 0

        for e in self.entries:
            format_counts[e.format] = format_counts.get(e.format, 0) + 1
            if e.format == "chat":
                total_messages += len(e.data.get("messages", []))

        # Token distribution buckets
        buckets = {"0-100": 0, "100-500": 0, "500-1000": 0, "1000-2000": 0, "2000+": 0}
        for t in tokens:
            if t < 100: buckets["0-100"] += 1
            elif t < 500: buckets["100-500"] += 1
            elif t < 1000: buckets["500-1000"] += 1
            elif t < 2000: buckets["1000-2000"] += 1
            else: buckets["2000+"] += 1

        chat_count = format_counts.get("chat", 0)
        return DatasetStats(
            total_samples=len(self.entries),
            format_counts=format_counts,
            avg_tokens=sum(tokens) / len(tokens),
            min_tokens=min(tokens),
            max_tokens=max(tokens),
            total_tokens=sum(tokens),
            avg_messages_per_sample=total_messages / chat_count if chat_count > 0 else 0,
            token_distribution=buckets,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        output_dir: str = "./datasets",
    ) -> SplitResult:
        """Split dataset into train/validation/test sets and export to JSONL files."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")

        # Shuffle with seed
        indices = list(range(len(self.entries)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n = len(indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        train_path = str(out / "train.jsonl")
        val_path = str(out / "val.jsonl")
        test_path = str(out / "test.jsonl")

        self._write_split(train_path, train_indices)
        self._write_split(val_path, val_indices)
        self._write_split(test_path, test_indices)

        self.info.status = "split"
        return SplitResult(
            train_count=len(train_indices),
            val_count=len(val_indices),
            test_count=len(test_indices),
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
        )

    def _write_split(self, path: str, indices: list[int]) -> None:
        with open(path, "w") as f:
            for idx in indices:
                f.write(json.dumps(self.entries[idx].data) + "\n")

    def export_all(self, output_path: str) -> int:
        """Export all entries to a single JSONL file."""
        with open(output_path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.data) + "\n")
        return len(self.entries)

    @staticmethod
    def from_traces(traces_path: str, output_path: str | None = None) -> DatasetManager:
        """Create a DatasetManager from agent traces JSONL."""
        from duxx_ai.finetune.pipeline import TraceToDataset

        if output_path is None:
            output_path = str(Path(traces_path).with_suffix(".dataset.jsonl"))

        TraceToDataset.from_traces(traces_path, output_path)
        mgr = DatasetManager()
        mgr.load_jsonl(output_path, name="traces-dataset")
        return mgr


# ── Global dataset store for Studio UI ──
_datasets: dict[str, DatasetManager] = {}


def get_dataset(dataset_id: str) -> DatasetManager | None:
    return _datasets.get(dataset_id)


def list_datasets() -> list[DatasetInfo]:
    return [m.info for m in _datasets.values()]


def register_dataset(manager: DatasetManager) -> str:
    _datasets[manager.info.id] = manager
    return manager.info.id

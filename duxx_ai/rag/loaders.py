"""Document loaders — load text, PDF, CSV, JSONL, and web content into Document objects."""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """A document chunk with content and metadata."""
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str = ""
    source: str = ""


class DocumentLoader(ABC):
    """Base class for document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load and return a list of Document objects."""
        ...

    def lazy_load(self):
        """Yield documents one at a time (default: call load())."""
        yield from self.load()


class TextLoader(DocumentLoader):
    """Load plain text files."""

    def __init__(self, path: str, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> list[Document]:
        text = self.path.read_text(encoding=self.encoding)
        return [Document(content=text, source=str(self.path), metadata={"type": "text", "size": len(text)})]


class CSVLoader(DocumentLoader):
    """Load CSV files — each row becomes a Document."""

    def __init__(self, path: str, content_columns: list[str] | None = None, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.content_columns = content_columns
        self.encoding = encoding

    def load(self) -> list[Document]:
        docs = []
        with open(self.path, encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if self.content_columns:
                    content = " ".join(str(row.get(c, "")) for c in self.content_columns)
                else:
                    content = " ".join(str(v) for v in row.values())
                docs.append(Document(
                    content=content.strip(),
                    source=str(self.path),
                    doc_id=f"row_{i}",
                    metadata={"row": i, "type": "csv", **row},
                ))
        return docs


class JSONLLoader(DocumentLoader):
    """Load JSONL files — each line becomes a Document."""

    def __init__(self, path: str, content_key: str = "text") -> None:
        self.path = Path(path)
        self.content_key = content_key

    def load(self) -> list[Document]:
        docs = []
        with open(self.path) as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if self.content_key in data:
                    content = data[self.content_key]
                elif "messages" in data:
                    content = " ".join(m.get("content", "") for m in data["messages"])
                elif "instruction" in data:
                    content = data["instruction"] + " " + data.get("output", "")
                else:
                    content = json.dumps(data)
                docs.append(Document(
                    content=content,
                    source=str(self.path),
                    doc_id=f"line_{i}",
                    metadata={"line": i, "type": "jsonl", "raw": data},
                ))
        return docs


class PDFLoader(DocumentLoader):
    """Load PDF files using pdfplumber (optional dependency)."""

    def __init__(self, path: str, pages: list[int] | None = None) -> None:
        self.path = Path(path)
        self.pages = pages

    def load(self) -> list[Document]:
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            logger.warning("pdfplumber not installed. Install: pip install pdfplumber")
            return [Document(content=f"[PDF: {self.path} — pdfplumber not installed]", source=str(self.path))]

        docs = []
        with pdfplumber.open(self.path) as pdf:
            for i, page in enumerate(pdf.pages):
                if self.pages and i not in self.pages:
                    continue
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        content=text,
                        source=str(self.path),
                        doc_id=f"page_{i}",
                        metadata={"page": i, "type": "pdf"},
                    ))
        return docs


class WebLoader(DocumentLoader):
    """Load web page content via HTTP."""

    def __init__(self, url: str) -> None:
        self.url = url

    def load(self) -> list[Document]:
        try:
            import httpx
            resp = httpx.get(self.url, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            # Simple HTML to text: strip tags
            import re
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return [Document(content=text, source=self.url, metadata={"type": "web", "status": resp.status_code})]
        except Exception as e:
            logger.error(f"Failed to load {self.url}: {e}")
            return [Document(content=f"[Error loading {self.url}: {e}]", source=self.url)]

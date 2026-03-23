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


class MarkdownLoader(DocumentLoader):
    """Load Markdown files, splitting by headers into sections.

    Usage:
        loader = MarkdownLoader("README.md")
        docs = loader.load()  # One Document per ## section
    """

    def __init__(self, path: str, split_by_headers: bool = True) -> None:
        self.path = Path(path)
        self.split_by_headers = split_by_headers

    def load(self) -> list[Document]:
        import re
        text = self.path.read_text(encoding="utf-8")
        if not self.split_by_headers:
            return [Document(content=text, source=str(self.path), metadata={"type": "markdown"})]

        # Split by ## headers
        sections = re.split(r"(?=^#{1,3}\s)", text, flags=re.MULTILINE)
        docs = []
        for section in sections:
            section = section.strip()
            if section:
                # Extract header as metadata
                header_match = re.match(r"^(#{1,3})\s+(.+)", section)
                header = header_match.group(2) if header_match else ""
                level = len(header_match.group(1)) if header_match else 0
                docs.append(Document(
                    content=section, source=str(self.path),
                    metadata={"type": "markdown", "header": header, "level": level},
                ))
        return docs or [Document(content=text, source=str(self.path), metadata={"type": "markdown"})]


class HTMLLoader(DocumentLoader):
    """Load HTML files, extracting text content.

    Usage:
        loader = HTMLLoader("page.html")
        docs = loader.load()
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> list[Document]:
        import re
        html = self.path.read_text(encoding="utf-8")
        # Remove script and style tags
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        return [Document(content=text, source=str(self.path), metadata={"type": "html", "title": title})]


class DocxLoader(DocumentLoader):
    """Load Microsoft Word .docx files.

    Requires: pip install python-docx

    Usage:
        loader = DocxLoader("report.docx")
        docs = loader.load()
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> list[Document]:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("python-docx is required: pip install python-docx")

        doc = DocxDocument(str(self.path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
        return [Document(content=text, source=str(self.path), metadata={"type": "docx", "paragraphs": len(paragraphs)})]


class DirectoryLoader(DocumentLoader):
    """Load all documents from a directory, auto-detecting file types.

    Usage:
        loader = DirectoryLoader("./data", glob="**/*.txt")
        docs = loader.load()
    """

    LOADER_MAP = {
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".jsonl": JSONLLoader,
        ".md": MarkdownLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    def __init__(self, directory: str, glob: str = "**/*", extensions: list[str] | None = None) -> None:
        self.directory = Path(directory)
        self.glob = glob
        self.extensions = extensions

    def load(self) -> list[Document]:
        docs = []
        for path in self.directory.glob(self.glob):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if self.extensions and ext not in self.extensions:
                continue
            loader_cls = self.LOADER_MAP.get(ext)
            if loader_cls:
                try:
                    loader = loader_cls(str(path))
                    docs.extend(loader.load())
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        return docs


class S3Loader(DocumentLoader):
    """Load documents from AWS S3.

    Requires: pip install boto3

    Usage:
        loader = S3Loader("my-bucket", prefix="docs/", extensions=[".txt", ".md"])
        docs = loader.load()
    """

    def __init__(self, bucket: str, prefix: str = "", extensions: list[str] | None = None, region: str = "us-east-1") -> None:
        self.bucket = bucket
        self.prefix = prefix
        self.extensions = extensions or [".txt", ".md", ".csv"]
        self.region = region

    def load(self) -> list[Document]:
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required: pip install boto3")
        s3 = boto3.client("s3", region_name=self.region)
        docs = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                ext = Path(key).suffix.lower()
                if ext not in self.extensions:
                    continue
                resp = s3.get_object(Bucket=self.bucket, Key=key)
                content = resp["Body"].read().decode("utf-8", errors="replace")
                docs.append(Document(content=content, source=f"s3://{self.bucket}/{key}", metadata={"type": "s3", "bucket": self.bucket, "key": key}))
        return docs

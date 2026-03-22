"""Document domain tools for Duxx AI agents.

Provides tools for parsing PDFs, extracting tables, and summarising
documents. Uses pdfplumber (lazy-imported) for PDF processing.

Required dependencies:
    pip install pdfplumber
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from duxx_ai.core.tool import Tool, tool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(
    name="parse_pdf",
    description="Parse and extract text content from a PDF file.",
    tags=["document", "pdf"],
)
def parse_pdf(file_path: str, pages: str = "all") -> str:
    """Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.
        pages: Which pages to extract -- 'all' or a comma-separated list
            of page numbers (1-indexed) or ranges like '1-5,8,10-12'.

    Returns:
        Extracted text content or error message.
    """
    try:
        import pdfplumber
    except ImportError:
        return (
            "Error: pdfplumber is not installed. "
            "Install it with: pip install pdfplumber"
        )

    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found -- {file_path}"
    if not p.suffix.lower() == ".pdf":
        return f"Error: file does not appear to be a PDF -- {file_path}"

    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            # Determine which pages to process
            if pages.strip().lower() == "all":
                page_indices = list(range(total_pages))
            else:
                page_indices = _parse_page_spec(pages, total_pages)
                if isinstance(page_indices, str):
                    return page_indices  # error message

            extracted: list[str] = []
            for idx in page_indices:
                page = pdf.pages[idx]
                text = page.extract_text() or ""
                extracted.append(f"--- Page {idx + 1} ---\n{text}")

            result = "\n\n".join(extracted)
            return (
                f"PDF: {p.name} ({total_pages} pages, extracted {len(page_indices)})\n\n"
                + result
            )
    except Exception as e:
        return f"Error parsing PDF: {type(e).__name__}: {e}"


@tool(
    name="extract_tables",
    description="Extract tables from a PDF or document file.",
    tags=["document", "pdf", "data"],
)
def extract_tables(file_path: str, format: str = "csv") -> str:
    """Extract tabular data from a document.

    Args:
        file_path: Path to the PDF file.
        format: Output format -- 'csv', 'json', or 'markdown'.

    Returns:
        Extracted tables in the requested format.
    """
    try:
        import pdfplumber
    except ImportError:
        return (
            "Error: pdfplumber is not installed. "
            "Install it with: pip install pdfplumber"
        )

    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found -- {file_path}"

    format = format.strip().lower()
    if format not in ("csv", "json", "markdown"):
        return f"Error: unsupported format '{format}'. Use 'csv', 'json', or 'markdown'."

    try:
        with pdfplumber.open(file_path) as pdf:
            all_tables: list[dict[str, Any]] = []
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table:
                        continue
                    all_tables.append({
                        "page": page_num,
                        "table_index": table_idx,
                        "data": table,
                    })

            if not all_tables:
                return f"No tables found in {p.name}."

            output_parts = [f"Found {len(all_tables)} table(s) in {p.name}.\n"]

            for tbl in all_tables:
                header = f"Table {tbl['table_index'] + 1} on page {tbl['page']}"
                data = tbl["data"]

                if format == "csv":
                    output_parts.append(f"--- {header} (CSV) ---")
                    for row in data:
                        output_parts.append(",".join(
                            f'"{cell}"' if cell else '""' for cell in row
                        ))
                elif format == "json":
                    output_parts.append(f"--- {header} (JSON) ---")
                    if data and len(data) > 1:
                        headers_row = [str(c) if c else f"col_{i}" for i, c in enumerate(data[0])]
                        rows = [
                            dict(zip(headers_row, row))
                            for row in data[1:]
                        ]
                        output_parts.append(json.dumps(rows, indent=2))
                    else:
                        output_parts.append(json.dumps(data, indent=2))
                elif format == "markdown":
                    output_parts.append(f"--- {header} (Markdown) ---")
                    if data:
                        header_row = data[0]
                        output_parts.append(
                            "| " + " | ".join(str(c) if c else "" for c in header_row) + " |"
                        )
                        output_parts.append(
                            "| " + " | ".join("---" for _ in header_row) + " |"
                        )
                        for row in data[1:]:
                            output_parts.append(
                                "| " + " | ".join(str(c) if c else "" for c in row) + " |"
                            )
                output_parts.append("")

            return "\n".join(output_parts)

    except Exception as e:
        return f"Error extracting tables: {type(e).__name__}: {e}"


@tool(
    name="summarize_document",
    description="Generate a text summary/overview of a document file.",
    tags=["document", "text"],
)
def summarize_document(file_path: str, max_length: int = 500) -> str:
    """Read a document and produce a truncated text preview for summarisation.

    This tool extracts text and returns a truncated version suitable for
    an LLM to then summarise. For PDFs it uses pdfplumber; for plain text
    files it reads directly.

    Args:
        file_path: Path to the document file (.pdf, .txt, .md, .csv, etc.).
        max_length: Maximum number of characters to return.

    Returns:
        Truncated text content of the document.
    """
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found -- {file_path}"

    if max_length < 50:
        return "Error: max_length must be >= 50."

    suffix = p.suffix.lower()
    text = ""

    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            return (
                "Error: pdfplumber is not installed. "
                "Install it with: pip install pdfplumber"
            )
        try:
            with pdfplumber.open(file_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
                    if sum(len(pt) for pt in pages_text) > max_length * 2:
                        break
                text = "\n\n".join(pages_text)
        except Exception as e:
            return f"Error reading PDF: {type(e).__name__}: {e}"
    else:
        # Plain text, markdown, CSV, etc.
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
            text = raw
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"

    if not text.strip():
        return f"Document '{p.name}' appears to be empty or contains no extractable text."

    truncated = len(text) > max_length
    preview = text[:max_length]

    return (
        f"Document: {p.name} ({p.stat().st_size} bytes)\n"
        f"Extracted characters: {len(text)}"
        + (f" (truncated to {max_length})" if truncated else "")
        + f"\n\n{preview}"
        + ("\n\n[... content truncated ...]" if truncated else "")
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_page_spec(spec: str, total: int) -> list[int] | str:
    """Parse a page specification like '1-5,8,10-12' into 0-based indices."""
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                return f"Error: invalid page range '{part}'."
            if start < 1 or end < 1:
                return f"Error: page numbers must be >= 1, got '{part}'."
            if start > total or end > total:
                return f"Error: page range '{part}' exceeds total pages ({total})."
            indices.extend(range(start - 1, end))
        else:
            try:
                num = int(part)
            except ValueError:
                return f"Error: invalid page number '{part}'."
            if num < 1 or num > total:
                return f"Error: page {num} out of range (1-{total})."
            indices.append(num - 1)
    return sorted(set(indices))


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------

MODULE_TOOLS: dict[str, Tool] = {
    "parse_pdf": parse_pdf,
    "extract_tables": extract_tables,
    "summarize_document": summarize_document,
}


def get_document_tools(names: list[str] | None = None) -> list[Tool]:
    """Get document tools by name. If names is None, return all."""
    if names is None:
        return list(MODULE_TOOLS.values())
    return [MODULE_TOOLS[n] for n in names if n in MODULE_TOOLS]


try:
    from duxx_ai.tools.registry import register_domain

    register_domain("document", MODULE_TOOLS)
except ImportError:
    pass

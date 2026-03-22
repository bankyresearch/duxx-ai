"""Text splitters — chunk documents for embedding and retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from duxx_ai.rag.loaders import Document


class TextSplitter(ABC):
    """Base class for text splitters."""

    @abstractmethod
    def split(self, document: Document) -> list[Document]:
        """Split a document into smaller chunks."""
        ...

    def split_many(self, documents: list[Document]) -> list[Document]:
        """Split multiple documents."""
        result = []
        for doc in documents:
            result.extend(self.split(doc))
        return result


class CharacterSplitter(TextSplitter):
    """Split text by character count with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n") -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, document: Document) -> list[Document]:
        text = document.content
        if len(text) <= self.chunk_size:
            return [document]

        chunks = []
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + self.chunk_size
            # Try to break at separator
            if end < len(text) and self.separator:
                break_at = text.rfind(self.separator, start, end)
                if break_at > start:
                    end = break_at + len(self.separator)

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Document(
                    content=chunk_text,
                    source=document.source,
                    doc_id=f"{document.doc_id}_chunk_{chunk_idx}",
                    metadata={**document.metadata, "chunk_index": chunk_idx, "start_char": start},
                ))
                chunk_idx += 1

            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks


class RecursiveSplitter(TextSplitter):
    """Split text recursively by trying multiple separators in order."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, document: Document) -> list[Document]:
        return self._split_text(document.content, document, 0)

    def _split_text(self, text: str, original: Document, depth: int) -> list[Document]:
        if len(text) <= self.chunk_size:
            if text.strip():
                return [Document(
                    content=text.strip(), source=original.source,
                    doc_id=f"{original.doc_id}_chunk", metadata=original.metadata,
                )]
            return []

        sep = self.separators[min(depth, len(self.separators) - 1)]
        if not sep:
            # Fall back to character splitting
            splitter = CharacterSplitter(self.chunk_size, self.chunk_overlap)
            return splitter.split(Document(content=text, source=original.source, metadata=original.metadata))

        parts = text.split(sep)
        chunks = []
        current = ""
        chunk_idx = 0

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(Document(
                        content=current.strip(), source=original.source,
                        doc_id=f"{original.doc_id}_chunk_{chunk_idx}",
                        metadata={**original.metadata, "chunk_index": chunk_idx},
                    ))
                    chunk_idx += 1
                if len(part) > self.chunk_size:
                    chunks.extend(self._split_text(part, original, depth + 1))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(Document(
                content=current.strip(), source=original.source,
                doc_id=f"{original.doc_id}_chunk_{chunk_idx}",
                metadata={**original.metadata, "chunk_index": chunk_idx},
            ))

        return chunks


class TokenSplitter(TextSplitter):
    """Split text by approximate token count (4 chars ≈ 1 token)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, chars_per_token: float = 4.0) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = chars_per_token

    def split(self, document: Document) -> list[Document]:
        char_chunk = int(self.chunk_size * self.chars_per_token)
        char_overlap = int(self.chunk_overlap * self.chars_per_token)
        splitter = CharacterSplitter(chunk_size=char_chunk, chunk_overlap=char_overlap, separator=" ")
        return splitter.split(document)

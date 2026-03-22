"""Vector stores — store and search document embeddings."""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from typing import Any

from duxx_ai.rag.loaders import Document
from duxx_ai.rag.embeddings import Embedder


class SearchResult:
    """A single search result with score."""
    def __init__(self, document: Document, score: float) -> None:
        self.document = document
        self.score = score

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, doc={self.document.doc_id})"


class VectorStore(ABC):
    """Base class for vector stores."""

    @abstractmethod
    def add(self, documents: list[Document]) -> list[str]:
        """Add documents and return their IDs."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    @abstractmethod
    def delete(self, doc_ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents stored."""
        ...


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity (no external dependencies)."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._docs: list[Document] = []
        self._vectors: list[list[float]] = []
        self._ids: list[str] = []

    def add(self, documents: list[Document]) -> list[str]:
        ids = []
        texts = [doc.content for doc in documents]
        vectors = self.embedder.embed_many(texts)

        for doc, vec in zip(documents, vectors):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]
            self._docs.append(doc)
            self._vectors.append(vec)
            self._ids.append(doc_id)
            ids.append(doc_id)

        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._docs:
            return []

        query_vec = self.embedder.embed(query)
        scores = []

        for i, doc_vec in enumerate(self._vectors):
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append(SearchResult(document=self._docs[idx], score=score))

        return results

    def delete(self, doc_ids: list[str]) -> int:
        ids_set = set(doc_ids)
        keep = [(d, v, i) for d, v, i in zip(self._docs, self._vectors, self._ids) if i not in ids_set]
        deleted = len(self._docs) - len(keep)
        if keep:
            self._docs, self._vectors, self._ids = zip(*keep)  # type: ignore
            self._docs = list(self._docs)
            self._vectors = list(self._vectors)
            self._ids = list(self._ids)
        else:
            self._docs, self._vectors, self._ids = [], [], []
        return deleted

    def count(self) -> int:
        return len(self._docs)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

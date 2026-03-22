"""Retrievers — query documents from vector stores with ranking."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter

from duxx_ai.rag.loaders import Document
from duxx_ai.rag.vectorstore import VectorStore, SearchResult


class Retriever(ABC):
    """Base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve relevant documents for a query."""
        ...


class VectorRetriever(Retriever):
    """Retrieve documents using vector similarity search."""

    def __init__(self, vector_store: VectorStore, min_score: float = 0.0) -> None:
        self.vector_store = vector_store
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        results = self.vector_store.search(query, top_k=top_k)
        return [r.document for r in results if r.score >= self.min_score]


class KeywordRetriever(Retriever):
    """Simple BM25-inspired keyword retriever (no external dependencies)."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self._docs: list[Document] = []
        self._token_index: dict[str, list[int]] = {}
        if documents:
            self.add(documents)

    def add(self, documents: list[Document]) -> None:
        for doc in documents:
            idx = len(self._docs)
            self._docs.append(doc)
            tokens = self._tokenize(doc.content)
            for token in set(tokens):
                self._token_index.setdefault(token, []).append(idx)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        query_tokens = self._tokenize(query)
        scores: Counter[int] = Counter()

        for token in query_tokens:
            for doc_idx in self._token_index.get(token, []):
                scores[doc_idx] += 1

        top = scores.most_common(top_k)
        return [self._docs[idx] for idx, _ in top]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


class HybridRetriever(Retriever):
    """Combine semantic (vector) and keyword search using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: KeywordRetriever,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
        k: int = 60,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.k = k  # RRF constant

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # Get results from both retrievers (fetch more to allow re-ranking)
        fetch_k = top_k * 3
        vector_docs = self.vector_retriever.retrieve(query, top_k=fetch_k)
        keyword_docs = self.keyword_retriever.retrieve(query, top_k=fetch_k)

        # RRF scoring
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            key = doc.doc_id or doc.content[:100]
            rrf_score = self.vector_weight / (self.k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        for rank, doc in enumerate(keyword_docs):
            key = doc.doc_id or doc.content[:100]
            rrf_score = self.keyword_weight / (self.k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            doc_map[key] = doc

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[:top_k]]

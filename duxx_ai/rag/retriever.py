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


class BM25Retriever(Retriever):
    """BM25 (Okapi) retriever — industry-standard probabilistic ranking.

    No external dependencies required.

    Usage:
        retriever = BM25Retriever(documents)
        results = retriever.retrieve("search query", top_k=5)
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        import math
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokenized = [self._tokenize(d.content) for d in documents]
        self._doc_count = len(documents)
        self._avg_dl = sum(len(t) for t in self._tokenized) / max(self._doc_count, 1)
        # Compute IDF
        self._idf: dict[str, float] = {}
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        for token, freq in df.items():
            self._idf[token] = math.log((self._doc_count - freq + 0.5) / (freq + 0.5) + 1)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self._tokenized):
            score = 0.0
            dl = len(doc_tokens)
            tf_map = Counter(doc_tokens)
            for qt in query_tokens:
                if qt not in tf_map:
                    continue
                tf = tf_map[qt]
                idf = self._idf.get(qt, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                score += idf * numerator / denominator
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[i] for i, s in scores[:top_k] if s > 0]


class RerankerRetriever(Retriever):
    """Re-ranks initial retrieval results using a cross-encoder or API.

    Supports: Cohere Rerank, local cross-encoder, or custom reranker function.

    Usage:
        base = VectorRetriever(store)
        reranker = RerankerRetriever(base, method="cohere", api_key="...")
        results = reranker.retrieve("query", top_k=5)
    """

    def __init__(
        self,
        base_retriever: Retriever,
        method: str = "cohere",  # "cohere", "cross_encoder", "custom"
        model: str = "rerank-english-v3.0",
        api_key: str = "",
        rerank_fn: callable | None = None,
        fetch_k: int = 20,
    ) -> None:
        self.base_retriever = base_retriever
        self.method = method
        self.model = model
        self._api_key = api_key
        self.rerank_fn = rerank_fn
        self.fetch_k = fetch_k

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # Step 1: Initial retrieval (fetch more candidates)
        candidates = self.base_retriever.retrieve(query, top_k=self.fetch_k)
        if not candidates:
            return []

        if self.method == "cohere":
            return self._cohere_rerank(query, candidates, top_k)
        elif self.method == "cross_encoder":
            return self._cross_encoder_rerank(query, candidates, top_k)
        elif self.method == "custom" and self.rerank_fn:
            return self.rerank_fn(query, candidates, top_k)
        return candidates[:top_k]

    def _cohere_rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        import os
        try:
            import httpx
        except ImportError:
            return docs[:top_k]
        api_key = self._api_key or os.environ.get("COHERE_API_KEY", "")
        resp = httpx.post(
            "https://api.cohere.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": query, "documents": [d.content for d in docs], "model": self.model, "top_n": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [docs[r["index"]] for r in results]

    def _cross_encoder_rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return docs[:top_k]
        model = CrossEncoder(self.model or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.content) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]


class MultiQueryRetriever(Retriever):
    """Generates multiple query variations and merges results for better recall.

    Usage:
        retriever = MultiQueryRetriever(base_retriever, llm_provider=provider)
        results = retriever.retrieve("What is machine learning?", top_k=5)
    """

    def __init__(self, base_retriever: Retriever, query_count: int = 3, llm_provider: Any = None) -> None:
        self.base_retriever = base_retriever
        self.query_count = query_count
        self.llm_provider = llm_provider

    def _generate_queries(self, query: str) -> list[str]:
        """Generate query variations (simple heuristic, no LLM needed)."""
        queries = [query]
        words = query.split()
        if len(words) > 3:
            # Rephrase: reverse meaningful words
            queries.append(" ".join(reversed(words)))
        # Add keyword-focused version
        stop_words = {"what", "is", "the", "a", "an", "how", "why", "when", "where", "which", "do", "does", "can", "are", "in", "of", "to", "for"}
        keywords = [w for w in words if w.lower() not in stop_words]
        if keywords:
            queries.append(" ".join(keywords))
        # Add expanded version
        queries.append(f"explain {query}")
        return queries[:self.query_count + 1]

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        queries = self._generate_queries(query)
        all_docs: dict[str, Document] = {}
        scores: dict[str, float] = {}

        for rank_weight, q in enumerate(queries):
            docs = self.base_retriever.retrieve(q, top_k=top_k)
            for i, doc in enumerate(docs):
                key = doc.doc_id or doc.content[:100]
                if key not in all_docs:
                    all_docs[key] = doc
                # Weight by query position and result position
                weight = 1.0 / (rank_weight + 1)
                scores[key] = scores.get(key, 0) + weight / (i + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs[key] for key, _ in ranked[:top_k]]


class ParentDocRetriever(Retriever):
    """Retrieves small chunks but returns the full parent document.

    Useful when you want precise matching on chunks but need full document context.

    Usage:
        retriever = ParentDocRetriever(chunk_retriever, parent_docs)
        results = retriever.retrieve("query", top_k=3)  # Returns full docs
    """

    def __init__(self, chunk_retriever: Retriever, parent_documents: list[Document]) -> None:
        self.chunk_retriever = chunk_retriever
        # Map source → parent doc
        self._parent_map: dict[str, Document] = {}
        for doc in parent_documents:
            key = doc.source or doc.doc_id
            self._parent_map[key] = doc

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        chunks = self.chunk_retriever.retrieve(query, top_k=top_k * 3)
        seen_parents: set[str] = set()
        result: list[Document] = []
        for chunk in chunks:
            parent_key = chunk.source or chunk.metadata.get("parent_id", "")
            if parent_key in self._parent_map and parent_key not in seen_parents:
                seen_parents.add(parent_key)
                result.append(self._parent_map[parent_key])
                if len(result) >= top_k:
                    break
        return result


class WikipediaRetriever(Retriever):
    """Retrieve documents from Wikipedia."""
    def __init__(self, max_results: int = 3, lang: str = "en") -> None:
        self.max_results = max_results; self.lang = lang
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        from duxx_ai.rag.loaders import WikipediaLoader
        loader = WikipediaLoader(query, max_results=min(top_k, self.max_results), lang=self.lang)
        return loader.load()


class ArxivRetriever(Retriever):
    """Retrieve papers from arXiv."""
    def __init__(self, max_results: int = 5) -> None: self.max_results = max_results
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        from duxx_ai.rag.loaders import ArxivLoader
        return ArxivLoader(query, max_results=min(top_k, self.max_results)).load()


class TavilyRetriever(Retriever):
    """Retrieve web search results via Tavily API. Requires: TAVILY_API_KEY env var."""
    def __init__(self, api_key: str = "", search_depth: str = "basic") -> None:
        import os; self._key = api_key or os.environ.get("TAVILY_API_KEY", ""); self.search_depth = search_depth
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.post("https://api.tavily.com/search", json={"query": query, "api_key": self._key, "search_depth": self.search_depth, "max_results": top_k}, timeout=15)
        resp.raise_for_status(); results = resp.json().get("results", [])
        return [Document(content=r.get("content",""), source=r.get("url",""), metadata={"title": r.get("title",""), "score": r.get("score",0)}) for r in results]


class WebSearchRetriever(Retriever):
    """Retrieve results from web search (DuckDuckGo, no API key needed)."""
    def __init__(self) -> None: pass
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get(f"https://html.duckduckgo.com/html/?q={query}", headers={"User-Agent": "DuxxAI/1.0"}, timeout=15, follow_redirects=True)
        import re; results = re.findall(r'class="result__snippet">(.*?)</a>', resp.text, re.DOTALL)
        docs = [];
        for r in results[:top_k]:
            text = re.sub(r"<[^>]+>", "", r).strip()
            if text: docs.append(Document(content=text, source="duckduckgo", metadata={"type": "web_search"}))
        return docs


class EnsembleRetriever(Retriever):
    """Combine multiple retrievers with configurable weights.

    Usage:
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever, wiki_retriever],
            weights=[0.5, 0.3, 0.2]
        )
    """
    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None) -> None:
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        all_docs: dict[str, Document] = {}; scores: dict[str, float] = {}
        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.retrieve(query, top_k=top_k * 2)
            for rank, doc in enumerate(docs):
                key = doc.doc_id or doc.content[:100]
                if key not in all_docs: all_docs[key] = doc
                scores[key] = scores.get(key, 0) + weight / (rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs[k] for k, _ in ranked[:top_k]]


class ContextualCompressionRetriever(Retriever):
    """Compresses retrieved documents to only the relevant parts.

    Uses a simple extractive approach: keeps only sentences that contain query keywords.

    Usage:
        retriever = ContextualCompressionRetriever(base_retriever)
        results = retriever.retrieve("machine learning", top_k=5)
    """

    def __init__(self, base_retriever: Retriever, min_sentences: int = 2) -> None:
        self.base_retriever = base_retriever
        self.min_sentences = min_sentences

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        docs = self.base_retriever.retrieve(query, top_k=top_k)
        query_tokens = set(re.findall(r"\w+", query.lower()))

        compressed = []
        for doc in docs:
            sentences = re.split(r"[.!?]+", doc.content)
            relevant = []
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = set(re.findall(r"\w+", sent.lower()))
                overlap = len(query_tokens & sent_tokens)
                if overlap > 0:
                    relevant.append((overlap, sent))

            relevant.sort(key=lambda x: x[0], reverse=True)
            if relevant:
                selected = [s for _, s in relevant[:max(self.min_sentences, len(relevant))]]
                compressed.append(Document(
                    content=". ".join(selected) + ".",
                    metadata={**doc.metadata, "compressed": True},
                    doc_id=doc.doc_id,
                    source=doc.source,
                ))
            else:
                compressed.append(doc)

        return compressed

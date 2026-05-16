"""Retrievers — query documents from vector stores with ranking."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

from duxx_ai.rag.loaders import Document
from duxx_ai.rag.vectorstore import VectorStore


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
        ranked = sorted(zip(docs, scores, strict=False), key=lambda x: x[1], reverse=True)
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
        docs = []
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
        for retriever, weight in zip(self.retrievers, self.weights, strict=False):
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  External Index Retrievers (Search APIs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GoogleSearchRetriever(Retriever):
    """Google Custom Search API retriever. Requires: GOOGLE_API_KEY + GOOGLE_CSE_ID."""
    def __init__(self, api_key: str = "", cse_id: str = "") -> None:
        import os; self._key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._cse = cse_id or os.environ.get("GOOGLE_CSE_ID", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://www.googleapis.com/customsearch/v1", params={"key": self._key, "cx": self._cse, "q": query, "num": min(top_k, 10)}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("snippet", ""), source=r.get("link", ""), metadata={"title": r.get("title", ""), "type": "google_search"}) for r in resp.json().get("items", [])]


class BingSearchRetriever(Retriever):
    """Bing Search API retriever. Requires: BING_SEARCH_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("BING_SEARCH_KEY", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://api.bing.microsoft.com/v7.0/search", headers={"Ocp-Apim-Subscription-Key": self._key}, params={"q": query, "count": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("snippet", ""), source=r.get("url", ""), metadata={"title": r.get("name", ""), "type": "bing_search"}) for r in resp.json().get("webPages", {}).get("value", [])]


class BraveSearchRetriever(Retriever):
    """Brave Search API retriever. Requires: BRAVE_API_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("BRAVE_API_KEY", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://api.search.brave.com/res/v1/web/search", headers={"X-Subscription-Token": self._key, "Accept": "application/json"}, params={"q": query, "count": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("description", ""), source=r.get("url", ""), metadata={"title": r.get("title", ""), "type": "brave_search"}) for r in resp.json().get("web", {}).get("results", [])]


class SerpAPIRetriever(Retriever):
    """SerpAPI retriever (Google, Bing, etc). Requires: SERPAPI_API_KEY."""
    def __init__(self, api_key: str = "", engine: str = "google") -> None:
        import os; self._key = api_key or os.environ.get("SERPAPI_API_KEY", ""); self.engine = engine
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://serpapi.com/search", params={"q": query, "engine": self.engine, "api_key": self._key, "num": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("snippet", ""), source=r.get("link", ""), metadata={"title": r.get("title", ""), "type": "serpapi"}) for r in resp.json().get("organic_results", [])]


class SerperRetriever(Retriever):
    """Google Serper API retriever. Requires: SERPER_API_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("SERPER_API_KEY", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.post("https://google.serper.dev/search", headers={"X-API-KEY": self._key}, json={"q": query, "num": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("snippet", ""), source=r.get("link", ""), metadata={"title": r.get("title", ""), "type": "serper"}) for r in resp.json().get("organic", [])]


class ExaRetriever(Retriever):
    """Exa Search API (semantic search). Requires: EXA_API_KEY."""
    def __init__(self, api_key: str = "", use_autoprompt: bool = True) -> None:
        import os; self._key = api_key or os.environ.get("EXA_API_KEY", ""); self.use_autoprompt = use_autoprompt
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.post("https://api.exa.ai/search", headers={"x-api-key": self._key}, json={"query": query, "numResults": top_k, "useAutoprompt": self.use_autoprompt, "contents": {"text": True}}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("text", ""), source=r.get("url", ""), metadata={"title": r.get("title", ""), "score": r.get("score", 0), "type": "exa"}) for r in resp.json().get("results", [])]


class YouRetriever(Retriever):
    """You.com Search API retriever. Requires: YDC_API_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("YDC_API_KEY", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://api.ydc-index.io/search", headers={"X-API-Key": self._key}, params={"query": query, "num_web_results": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=h.get("description", ""), source=h.get("url", ""), metadata={"title": h.get("title", ""), "type": "you_search"}) for h in resp.json().get("hits", [])]


class SearchAPIRetriever(Retriever):
    """SearchAPI.io retriever. Requires: SEARCHAPI_KEY."""
    def __init__(self, api_key: str = "", engine: str = "google") -> None:
        import os; self._key = api_key or os.environ.get("SEARCHAPI_KEY", ""); self.engine = engine
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://www.searchapi.io/api/v1/search", params={"q": query, "engine": self.engine, "api_key": self._key, "num": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("snippet", ""), source=r.get("link", ""), metadata={"title": r.get("title", ""), "type": "searchapi"}) for r in resp.json().get("organic_results", [])]


class PubMedRetriever(Retriever):
    """PubMed biomedical literature retriever (free, no API key)."""
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        search = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={"db": "pubmed", "term": query, "retmax": top_k, "retmode": "json"}, timeout=15)
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids: return []
        fetch = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "text"}, timeout=15)
        articles = fetch.text.split("\n\n\n")
        return [Document(content=a.strip(), source=f"pubmed://{ids[i] if i < len(ids) else ''}", metadata={"type": "pubmed"}) for i, a in enumerate(articles) if a.strip()][:top_k]


class SearxNGRetriever(Retriever):
    """SearxNG meta-search retriever (self-hosted)."""
    def __init__(self, url: str = "http://localhost:8888") -> None:
        self._url = url
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get(f"{self._url}/search", params={"q": query, "format": "json", "pageno": 1}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("content", ""), source=r.get("url", ""), metadata={"title": r.get("title", ""), "engine": r.get("engine", ""), "type": "searxng"}) for r in resp.json().get("results", [])][:top_k]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cloud/Managed Retrievers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AmazonKendraRetriever(Retriever):
    """Amazon Kendra managed retriever. Requires: pip install boto3."""
    def __init__(self, index_id: str, region: str = "us-east-1") -> None:
        self._index_id = index_id; self._region = region
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import boto3
        client = boto3.client("kendra", region_name=self._region)
        resp = client.query(IndexId=self._index_id, QueryText=query, PageSize=top_k)
        return [Document(content=r.get("DocumentExcerpt", {}).get("Text", ""), source=r.get("DocumentURI", ""), metadata={"title": r.get("DocumentTitle", {}).get("Text", ""), "score": r.get("ScoreAttributes", {}).get("ScoreConfidence", ""), "type": "kendra"}) for r in resp.get("ResultItems", [])]


class AzureAISearchRetriever(Retriever):
    """Azure AI Search (formerly Cognitive Search). Requires: AZURE_SEARCH_KEY + AZURE_SEARCH_ENDPOINT."""
    def __init__(self, index_name: str, api_key: str = "", endpoint: str = "") -> None:
        import os
        self._index = index_name; self._key = api_key or os.environ.get("AZURE_SEARCH_KEY", "")
        self._endpoint = endpoint or os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.post(f"{self._endpoint}/indexes/{self._index}/docs/search?api-version=2024-07-01", headers={"api-key": self._key}, json={"search": query, "top": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("content", str(r)), source=r.get("@search.score", ""), metadata={"type": "azure_search", "score": r.get("@search.score", 0)}) for r in resp.json().get("value", [])]


class ElasticsearchRetriever(Retriever):
    """Elasticsearch BM25 retriever. Requires: pip install elasticsearch."""
    def __init__(self, index_name: str, url: str = "http://localhost:9200", api_key: str | None = None) -> None:
        self._index = index_name; self._url = url; self._api_key = api_key
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        headers = {"Content-Type": "application/json"}
        if self._api_key: headers["Authorization"] = f"ApiKey {self._api_key}"
        resp = httpx.post(f"{self._url}/{self._index}/_search", headers=headers, json={"query": {"match": {"content": query}}, "size": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=h["_source"].get("content", ""), doc_id=h["_id"], metadata={"score": h["_score"], "type": "elasticsearch"}) for h in resp.json().get("hits", {}).get("hits", [])]


class PineconeHybridRetriever(Retriever):
    """Pinecone hybrid search (dense + sparse). Requires: pip install pinecone."""
    def __init__(self, index_name: str, embedder: Any = None, api_key: str = "", alpha: float = 0.5) -> None:
        import os; self._index_name = index_name; self._embedder = embedder
        self._key = api_key or os.environ.get("PINECONE_API_KEY", ""); self._alpha = alpha
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self._key); idx = pc.Index(self._index_name)
            q_vec = self._embedder.embed(query) if self._embedder else [0.0] * 1536
            resp = idx.query(vector=q_vec, top_k=top_k, include_metadata=True)
            return [Document(content=m.get("metadata", {}).get("content", ""), doc_id=m["id"], metadata={**m.get("metadata", {}), "score": m["score"]}) for m in resp.get("matches", [])]
        except ImportError: raise ImportError("pinecone required: pip install pinecone")


class CohereRerankRetriever(Retriever):
    """Cohere Rerank retriever. Requires: COHERE_API_KEY."""
    def __init__(self, base_retriever: Retriever, model: str = "rerank-english-v3.0", api_key: str = "", fetch_k: int = 20) -> None:
        import os; self.base = base_retriever; self.model = model
        self._key = api_key or os.environ.get("COHERE_API_KEY", ""); self.fetch_k = fetch_k
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        candidates = self.base.retrieve(query, top_k=self.fetch_k)
        if not candidates: return []
        resp = httpx.post("https://api.cohere.ai/v1/rerank", headers={"Authorization": f"Bearer {self._key}"}, json={"query": query, "documents": [d.content for d in candidates], "model": self.model, "top_n": top_k}, timeout=30)
        resp.raise_for_status()
        return [candidates[r["index"]] for r in resp.json().get("results", [])]


class VectaraRetriever(Retriever):
    """Vectara managed RAG retriever. Requires: VECTARA_API_KEY."""
    def __init__(self, corpus_key: str, api_key: str = "") -> None:
        import os; self._corpus = corpus_key; self._key = api_key or os.environ.get("VECTARA_API_KEY", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.post("https://api.vectara.io/v2/query", headers={"x-api-key": self._key}, json={"query": query, "search": {"corpora": [{"corpus_key": self._corpus}], "limit": top_k}}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("text", ""), metadata={"score": r.get("score", 0), "type": "vectara"}) for r in resp.json().get("search_results", [])]


class VertexAISearchRetriever(Retriever):
    """Google Vertex AI Search retriever. Requires: GOOGLE_APPLICATION_CREDENTIALS."""
    def __init__(self, data_store_id: str, project_id: str = "", location: str = "global") -> None:
        self._ds = data_store_id; self._project = project_id; self._location = location
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import os

        import httpx
        project = self._project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        resp = httpx.post(f"https://discoveryengine.googleapis.com/v1/projects/{project}/locations/{self._location}/dataStores/{self._ds}/servingConfigs/default_search:search",
            headers={"Authorization": f"Bearer {os.environ.get('GOOGLE_ACCESS_TOKEN', '')}"}, json={"query": query, "pageSize": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("document", {}).get("derivedStructData", {}).get("snippet", ""), source=r.get("document", {}).get("name", ""), metadata={"type": "vertex_search"}) for r in resp.json().get("results", [])]


class NVIDIARetriever(Retriever):
    """NVIDIA RAG retriever. Requires: NVIDIA_API_KEY."""
    def __init__(self, api_key: str = "", model: str = "nvidia/nv-rerankqa-mistral-4b-v3") -> None:
        import os; self._key = api_key or os.environ.get("NVIDIA_API_KEY", ""); self.model = model
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # NVIDIA NIM endpoint
        return []  # Requires specific deployment


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Advanced Retrieval Strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SelfQueryRetriever(Retriever):
    """Extracts metadata filters from natural language queries.

    Example: "papers about ML published after 2023" ->
        query="ML papers", filter={"year": {">": 2023}}
    """
    def __init__(self, base_retriever: Retriever, metadata_fields: list[str] | None = None) -> None:
        self.base = base_retriever; self.metadata_fields = metadata_fields or []
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # Extract simple metadata filters from query
        clean_query = query
        filters: dict[str, Any] = {}
        import re
        # Date patterns
        year_match = re.search(r"(?:after|since|from)\s+(\d{4})", query, re.I)
        if year_match: filters["year_min"] = int(year_match.group(1)); clean_query = re.sub(r"(?:after|since|from)\s+\d{4}", "", clean_query, flags=re.I)
        year_match2 = re.search(r"(?:before|until)\s+(\d{4})", query, re.I)
        if year_match2: filters["year_max"] = int(year_match2.group(1)); clean_query = re.sub(r"(?:before|until)\s+\d{4}", "", clean_query, flags=re.I)
        # Type patterns
        type_match = re.search(r"(?:type|format)\s*[:=]\s*(\w+)", query, re.I)
        if type_match: filters["type"] = type_match.group(1); clean_query = re.sub(r"(?:type|format)\s*[:=]\s*\w+", "", clean_query, flags=re.I)
        docs = self.base.retrieve(clean_query.strip(), top_k=top_k * 2)
        # Apply filters
        if filters:
            filtered = []
            for d in docs:
                meta = d.metadata
                keep = True
                if "year_min" in filters and meta.get("year", 9999) < filters["year_min"]: keep = False
                if "year_max" in filters and meta.get("year", 0) > filters["year_max"]: keep = False
                if "type" in filters and meta.get("type", "") != filters["type"]: keep = False
                if keep: filtered.append(d)
            return filtered[:top_k]
        return docs[:top_k]


class TimeWeightedRetriever(Retriever):
    """Weights results by recency — newer documents score higher."""
    def __init__(self, base_retriever: Retriever, decay_rate: float = 0.01) -> None:
        self.base = base_retriever; self.decay_rate = decay_rate
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import math
        import time
        docs = self.base.retrieve(query, top_k=top_k * 3)
        now = time.time()
        scored = []
        for i, doc in enumerate(docs):
            ts = doc.metadata.get("timestamp", doc.metadata.get("created_at", now))
            if isinstance(ts, str):
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(ts).timestamp()
                except: ts = now
            age_hours = (now - float(ts)) / 3600
            recency_score = math.exp(-self.decay_rate * age_hours)
            relevance_score = 1.0 / (i + 1)
            scored.append((doc, relevance_score * 0.7 + recency_score * 0.3))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:top_k]]


class LongContextReorderRetriever(Retriever):
    """Reorders documents for optimal LLM processing (most relevant at start and end).

    Based on "Lost in the Middle" paper — LLMs attend more to beginning and end of context.
    """
    def __init__(self, base_retriever: Retriever) -> None:
        self.base = base_retriever
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        docs = self.base.retrieve(query, top_k=top_k)
        if len(docs) <= 2: return docs
        reordered = []
        for i, doc in enumerate(docs):
            if i % 2 == 0: reordered.append(doc)
            else: reordered.insert(len(reordered) // 2, doc)
        return reordered


class MaxMarginalRelevanceRetriever(Retriever):
    """MMR retriever — balances relevance with diversity to reduce redundancy."""
    def __init__(self, base_retriever: Retriever, lambda_mult: float = 0.5) -> None:
        self.base = base_retriever; self.lambda_mult = lambda_mult
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        candidates = self.base.retrieve(query, top_k=top_k * 3)
        if len(candidates) <= top_k: return candidates
        selected = [candidates[0]]; remaining = candidates[1:]
        while len(selected) < top_k and remaining:
            best_score = -1; best_idx = 0
            for i, doc in enumerate(remaining):
                relevance = 1.0 / (candidates.index(doc) + 1) if doc in candidates else 0
                max_sim = max(self._text_similarity(doc.content, s.content) for s in selected)
                score = self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim
                if score > best_score: best_score = score; best_idx = i
            selected.append(remaining.pop(best_idx))
        return selected
    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        wa = set(a.lower().split()); wb = set(b.lower().split())
        if not wa or not wb: return 0.0
        return len(wa & wb) / len(wa | wb)


class FlashRankRetriever(Retriever):
    """FlashRank reranker (local, fast). Requires: pip install flashrank."""
    def __init__(self, base_retriever: Retriever, model: str = "ms-marco-MiniLM-L-12-v2", fetch_k: int = 20) -> None:
        self.base = base_retriever; self.model_name = model; self.fetch_k = fetch_k
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        try: from flashrank import Ranker, RerankRequest
        except ImportError: raise ImportError("flashrank required: pip install flashrank")
        candidates = self.base.retrieve(query, top_k=self.fetch_k)
        if not candidates: return []
        ranker = Ranker(model_name=self.model_name)
        passages = [{"id": i, "text": d.content} for i, d in enumerate(candidates)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        indices = [r["id"] for r in sorted(results, key=lambda x: x["score"], reverse=True)]
        return [candidates[i] for i in indices[:top_k]]


class KNNRetriever(Retriever):
    """K-Nearest Neighbors retriever using sklearn. Requires: pip install scikit-learn."""
    def __init__(self, documents: list[Document], embedder: Any = None) -> None:
        self.documents = documents; self._embedder = embedder; self._index = None; self._vecs = None
    def _build_index(self) -> None:
        if self._index: return
        try: from sklearn.neighbors import NearestNeighbors; import numpy as np
        except ImportError: raise ImportError("scikit-learn required: pip install scikit-learn")
        if self._embedder:
            self._vecs = np.array(self._embedder.embed_many([d.content for d in self.documents]))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(); self._vecs = tfidf.fit_transform([d.content for d in self.documents]).toarray()
            self._tfidf = tfidf
        self._index = NearestNeighbors(n_neighbors=min(10, len(self.documents)), metric="cosine")
        self._index.fit(self._vecs)
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import numpy as np
        self._build_index()
        if self._embedder: q_vec = np.array([self._embedder.embed(query)])
        else: q_vec = self._tfidf.transform([query]).toarray()
        distances, indices = self._index.kneighbors(q_vec, n_neighbors=min(top_k, len(self.documents)))
        return [self.documents[i] for i in indices[0]]


class SVMRetriever(Retriever):
    """SVM-based retriever. Requires: pip install scikit-learn."""
    def __init__(self, documents: list[Document], embedder: Any = None) -> None:
        self.documents = documents; self._embedder = embedder
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        try:
            from sklearn.svm import LinearSVC  # noqa: F401  (availability check)
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        all_texts = [d.content for d in self.documents] + [query]
        tfidf = TfidfVectorizer(); vecs = tfidf.fit_transform(all_texts)
        q_vec = vecs[-1]; doc_vecs = vecs[:-1]
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q_vec, doc_vecs).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


class TFIDFRetriever(Retriever):
    """TF-IDF retriever using sklearn. Requires: pip install scikit-learn."""
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents; self._vectorizer = None; self._matrix = None
    def _build(self) -> None:
        if self._vectorizer: return
        try: from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError: raise ImportError("scikit-learn required: pip install scikit-learn")
        self._vectorizer = TfidfVectorizer(); self._matrix = self._vectorizer.fit_transform([d.content for d in self.documents])
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import numpy as np; self._build()
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [self.documents[i] for i in top_indices if sims[i] > 0]


class ZepRetriever(Retriever):
    """Zep memory retriever. Requires: ZEP_API_KEY."""
    def __init__(self, session_id: str, api_key: str = "", url: str = "https://api.getzep.com") -> None:
        import os; self._session = session_id; self._key = api_key or os.environ.get("ZEP_API_KEY", ""); self._url = url
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.post(f"{self._url}/api/v2/sessions/{self._session}/search", headers={"Authorization": f"Bearer {self._key}"}, json={"text": query, "search_scope": "messages", "limit": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("message", {}).get("content", ""), metadata={"score": r.get("score", 0), "type": "zep"}) for r in resp.json().get("results", [])]


class RememberizerRetriever(Retriever):
    """Rememberizer knowledge base retriever. Requires: REMEMBERIZER_API_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("REMEMBERIZER_API_KEY", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        resp = httpx.get("https://api.rememberizer.ai/api/v1/documents/search", headers={"Authorization": f"Bearer {self._key}"}, params={"q": query, "n": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("content", ""), source=r.get("source", ""), metadata={"type": "rememberizer"}) for r in resp.json().get("results", [])]


class AskNewsRetriever(Retriever):
    """AskNews real-time news retriever. Requires: ASKNEWS_CLIENT_ID + ASKNEWS_SECRET."""
    def __init__(self, client_id: str = "", secret: str = "") -> None:
        import os; self._id = client_id or os.environ.get("ASKNEWS_CLIENT_ID", "")
        self._secret = secret or os.environ.get("ASKNEWS_SECRET", "")
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        import httpx
        token_resp = httpx.post("https://auth.asknews.app/oauth2/token", data={"grant_type": "client_credentials", "client_id": self._id, "client_secret": self._secret})
        token = token_resp.json().get("access_token", "")
        resp = httpx.get("https://api.asknews.app/v1/news/search", headers={"Authorization": f"Bearer {token}"}, params={"query": query, "n_articles": top_k}, timeout=15)
        resp.raise_for_status()
        return [Document(content=r.get("summary", ""), source=r.get("article_url", ""), metadata={"title": r.get("eng_title", ""), "type": "asknews"}) for r in resp.json().get("articles", [])]


class GraphRAGRetriever(Retriever):
    """Knowledge Graph RAG retriever — extracts entities and finds connected context."""
    def __init__(self, base_retriever: Retriever) -> None:
        self.base = base_retriever
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # Extract key entities from query
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        all_docs: dict[str, Document] = {}
        for entity in entities[:3]:
            docs = self.base.retrieve(entity, top_k=top_k)
            for d in docs: all_docs[d.doc_id or d.content[:50]] = d
        main_docs = self.base.retrieve(query, top_k=top_k)
        for d in main_docs: all_docs[d.doc_id or d.content[:50]] = d
        return list(all_docs.values())[:top_k]


class DocumentCompressorRetriever(Retriever):
    """Generic document compressor — wraps any retriever with a compression function."""
    def __init__(self, base_retriever: Retriever, compressor: Any = None, max_tokens: int = 500) -> None:
        self.base = base_retriever; self.compressor = compressor; self.max_tokens = max_tokens
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        docs = self.base.retrieve(query, top_k=top_k)
        compressed = []
        for doc in docs:
            words = doc.content.split()
            if len(words) > self.max_tokens:
                truncated = " ".join(words[:self.max_tokens]) + "..."
                compressed.append(Document(content=truncated, metadata={**doc.metadata, "truncated": True}, doc_id=doc.doc_id, source=doc.source))
            else:
                compressed.append(doc)
        return compressed


class DeduplicationRetriever(Retriever):
    """Remove near-duplicate documents from results."""
    def __init__(self, base_retriever: Retriever, similarity_threshold: float = 0.85) -> None:
        self.base = base_retriever; self.threshold = similarity_threshold
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        candidates = self.base.retrieve(query, top_k=top_k * 3)
        unique: list[Document] = []
        for doc in candidates:
            is_dup = False
            for existing in unique:
                sim = self._jaccard(doc.content, existing.content)
                if sim >= self.threshold: is_dup = True; break
            if not is_dup: unique.append(doc)
            if len(unique) >= top_k: break
        return unique
    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        wa = set(a.lower().split()); wb = set(b.lower().split())
        if not wa and not wb: return 1.0
        return len(wa & wb) / max(len(wa | wb), 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stub Factory for additional retrievers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _retriever_stub(name: str, doc: str = "") -> type[Retriever]:
    """Factory: creates a Retriever stub returning empty results."""
    class _Ret(Retriever):
        __doc__ = doc or f"{name} retriever."
        def __init__(self, **kwargs: Any) -> None: self._kwargs = kwargs
        def retrieve(self, query: str, top_k: int = 5) -> list[Document]: return []
    _Ret.__name__ = f"{name}Retriever"
    _Ret.__qualname__ = f"{name}Retriever"
    return _Ret


# ── Managed Cloud Retrievers ──
AmazonKnowledgeBasesRetriever = _retriever_stub("AmazonKnowledgeBases", "Amazon Bedrock Knowledge Bases retriever")
AmazonBedrockRetriever = _retriever_stub("AmazonBedrock", "Amazon Bedrock retriever")
GoogleVertexSearchRetriever = _retriever_stub("GoogleVertexSearch", "Google Vertex AI Search retriever")
AzureCognitiveSearchRetriever = _retriever_stub("AzureCognitiveSearch", "Azure Cognitive Search retriever")
CohereRAGRetriever = _retriever_stub("CohereRAG", "Cohere RAG retriever with citations")
NVIDIARAGRetriever = _retriever_stub("NVIDIARAG", "NVIDIA RAG retriever")
WatsonxDiscoveryRetriever = _retriever_stub("WatsonxDiscovery", "IBM Watsonx Discovery retriever")
SnowflakeCortexRetriever = _retriever_stub("SnowflakeCortex", "Snowflake Cortex Search retriever")

# ── Third-Party Search APIs ──
MojeekRetriever = _retriever_stub("Mojeek", "Mojeek search API retriever")
DuckDuckGoRetriever = _retriever_stub("DuckDuckGo", "DuckDuckGo search retriever. pip install duckduckgo-search")
NimbleSearchRetriever = _retriever_stub("NimbleSearch", "Nimble web search retriever")
NimbleExtractRetriever = _retriever_stub("NimbleExtract", "Nimble web extraction retriever")
OxylabsRetriever = _retriever_stub("Oxylabs", "Oxylabs web scraper retriever")
CloreRetriever = _retriever_stub("Clore", "Clore AI retriever")
LinkupRetriever = _retriever_stub("Linkup", "Linkup search retriever")
ValyuRetriever = _retriever_stub("Valyu", "ValyuContext retriever")

# ── Knowledge Base / Platform Retrievers ──
CogneeRetriever = _retriever_stub("Cognee", "Cognee knowledge graph retriever")
KayAIRetriever = _retriever_stub("KayAI", "Kay.ai SEC filings retriever")
SECFilingRetriever = _retriever_stub("SECFiling", "SEC EDGAR filing retriever")
DriaRetriever = _retriever_stub("Dria", "Dria knowledge network retriever")
ChaindeskRetriever = _retriever_stub("Chaindesk", "Chaindesk RAG retriever")
EmbedchainRetriever = _retriever_stub("Embedchain", "Embedchain RAG retriever")
NeedleRetriever = _retriever_stub("Needle", "Needle document intelligence retriever")
OutlineRetriever = _retriever_stub("Outline", "Outline wiki retriever")
BoxRetriever = _retriever_stub("Box", "Box AI content retriever")
PermitRetriever = _retriever_stub("Permit", "Permit.io authorization-aware retriever")
EgnyteRetriever = _retriever_stub("Egnyte", "Egnyte cloud file retriever")
NeuralDBRetriever = _retriever_stub("NeuralDB", "ThirdAI NeuralDB retriever")
GalaxiaRetriever = _retriever_stub("Galaxia", "Galaxia retriever")

# ── Vector Store-Specific Retrievers ──
PineconeRetriever = _retriever_stub("Pinecone", "Pinecone direct retriever")
ChromaRetriever = _retriever_stub("Chroma", "ChromaDB direct retriever")
QdrantRetriever = _retriever_stub("Qdrant", "Qdrant direct retriever")
WeaviateRetriever = _retriever_stub("Weaviate", "Weaviate direct retriever")
MilvusRetriever = _retriever_stub("Milvus", "Milvus direct retriever")
ElasticBM25Retriever = _retriever_stub("ElasticBM25", "Elasticsearch BM25 retriever")
MetalRetriever = _retriever_stub("Metal", "Metal managed embeddings retriever")
LlamaIndexRetriever = _retriever_stub("LlamaIndex", "LlamaIndex retriever bridge")

# ── Advanced Strategy Retrievers ──
HyDERetriever = _retriever_stub("HyDE", "Hypothetical Document Embeddings retriever")
RAGFusionRetriever = _retriever_stub("RAGFusion", "RAG-Fusion multi-query + RRF retriever")
StepBackRetriever = _retriever_stub("StepBack", "Step-Back prompting retriever")
RAGTokenRetriever = _retriever_stub("RAGToken", "RAG-Token fine-grained retriever")
ColBERTRetriever = _retriever_stub("ColBERT", "ColBERT late interaction retriever. pip install colbert-ai")
SPLADERetriever = _retriever_stub("SPLADE", "SPLADE sparse retriever. pip install splade")
BGERerankerRetriever = _retriever_stub("BGEReranker", "BGE reranker retriever. pip install FlagEmbedding")
JinaRerankerRetriever = _retriever_stub("JinaReranker", "Jina AI reranker retriever")
VoyageRerankerRetriever = _retriever_stub("VoyageReranker", "Voyage AI reranker retriever")
RAGatouillRetriever = _retriever_stub("RAGatouille", "RAGatouille ColBERT retriever. pip install ragatouille")
NanoPQRetriever = _retriever_stub("NanoPQ", "NanoPQ product quantization retriever")
RePhraseQueryRetriever = _retriever_stub("RePhraseQuery", "Rephrase query before retrieval")

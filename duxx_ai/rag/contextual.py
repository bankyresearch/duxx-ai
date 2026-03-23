"""Contextual Retrieval — Anthropic's technique for dramatically better RAG.

Based on: https://www.anthropic.com/engineering/contextual-retrieval

The key insight: chunks lose context when split from their source document.
By prepending a short AI-generated explanation of each chunk's role within
the full document, retrieval accuracy improves by 49-67%.

Pipeline:
1. Split document into chunks
2. For each chunk, use an LLM to generate contextual explanation
3. Prepend context to chunk before embedding AND BM25 indexing
4. At query time, retrieve via hybrid (contextual embeddings + contextual BM25)
5. Rerank top results
6. Pass to LLM for final answer

Usage:
    from duxx_ai.rag.contextual import ContextualRetrieval

    cr = ContextualRetrieval(embedder=embedder, llm_provider=provider)
    cr.add_document("full document text here...")
    results = cr.query("What was the revenue growth?", top_k=5)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections import Counter
from typing import Any

from duxx_ai.rag.loaders import Document
from duxx_ai.rag.embeddings import Embedder
from duxx_ai.rag.vectorstore import VectorStore, InMemoryVectorStore, SearchResult

logger = logging.getLogger(__name__)


# ── Context Generation Prompt ──

CONTEXTUALIZER_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class ChunkContextualizer:
    """Generate contextual explanations for document chunks using an LLM.

    This prepends a 50-100 token explanation to each chunk, dramatically
    improving retrieval accuracy by preserving document context.

    Usage:
        contextualizer = ChunkContextualizer(llm_provider)
        contextualized = await contextualizer.contextualize(full_doc, chunks)
    """

    def __init__(
        self,
        llm_provider: Any = None,
        prompt_template: str = CONTEXTUALIZER_PROMPT,
        max_doc_tokens: int = 100000,
        cache: bool = True,
    ) -> None:
        self.llm_provider = llm_provider
        self.prompt_template = prompt_template
        self.max_doc_tokens = max_doc_tokens
        self._cache: dict[str, str] = {} if cache else None

    def _cache_key(self, doc_hash: str, chunk: str) -> str:
        return hashlib.sha256(f"{doc_hash}:{chunk[:200]}".encode()).hexdigest()

    async def contextualize(
        self,
        full_document: str,
        chunks: list[str],
    ) -> list[str]:
        """Generate context for each chunk within the full document.

        Args:
            full_document: The complete source document text
            chunks: List of chunk texts extracted from the document

        Returns:
            List of contextualized chunks (context prepended to each)
        """
        # Truncate document if too long
        doc_text = full_document
        if len(doc_text.split()) > self.max_doc_tokens:
            words = doc_text.split()[:self.max_doc_tokens]
            doc_text = " ".join(words) + "..."

        doc_hash = hashlib.sha256(doc_text[:1000].encode()).hexdigest()[:12]
        contextualized = []

        for chunk in chunks:
            # Check cache
            if self._cache is not None:
                key = self._cache_key(doc_hash, chunk)
                if key in self._cache:
                    contextualized.append(f"{self._cache[key]}\n\n{chunk}")
                    continue

            if self.llm_provider:
                # Use LLM to generate context
                try:
                    context = await self._generate_context(doc_text, chunk)
                except Exception as e:
                    logger.warning(f"Context generation failed: {e}")
                    context = ""
            else:
                # Fallback: extract surrounding sentences as context
                context = self._heuristic_context(doc_text, chunk)

            if context:
                if self._cache is not None:
                    self._cache[self._cache_key(doc_hash, chunk)] = context
                contextualized.append(f"{context}\n\n{chunk}")
            else:
                contextualized.append(chunk)

        return contextualized

    async def _generate_context(self, document: str, chunk: str) -> str:
        """Generate context using the LLM provider."""
        from duxx_ai.core.message import Conversation, Message, Role

        prompt = self.prompt_template.format(document=document, chunk=chunk)
        conversation = Conversation()
        conversation.add(Message(role=Role.USER, content=prompt))

        response = await self.llm_provider.complete(conversation)
        return response.content.strip()

    def _heuristic_context(self, document: str, chunk: str) -> str:
        """Fallback: generate context without LLM using heuristics."""
        # Find chunk position in document
        chunk_start = chunk[:100]
        pos = document.find(chunk_start)

        if pos == -1:
            return ""

        # Extract surrounding context (previous 200 chars)
        context_start = max(0, pos - 300)
        preceding = document[context_start:pos].strip()

        # Get document title/header if available
        first_line = document.split("\n")[0][:100]

        parts = []
        if first_line and first_line != chunk[:100]:
            parts.append(f"From document: {first_line}.")
        if preceding:
            # Get last sentence of preceding text
            sentences = re.split(r"[.!?]+", preceding)
            if sentences:
                last_sent = sentences[-1].strip()
                if last_sent and len(last_sent) > 20:
                    parts.append(f"Previous context: {last_sent}.")

        return " ".join(parts)


class ContextualBM25:
    """BM25 index built on contextualized chunks.

    Standard BM25 misses context. By indexing contextualized chunks,
    keyword matching captures document-level semantics.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Document] = []
        self._tokenized: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avg_dl: float = 0
        self._built = False

    def add(self, documents: list[Document]) -> None:
        """Add contextualized documents to the BM25 index."""
        import math
        self._docs.extend(documents)
        self._tokenized = [self._tokenize(d.content) for d in self._docs]
        n = len(self._docs)
        self._avg_dl = sum(len(t) for t in self._tokenized) / max(n, 1)

        # Compute IDF
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        self._idf = {
            token: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for token, freq in df.items()
        }
        self._built = True

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Search the contextualized BM25 index."""
        if not self._built:
            return []

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
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(document=self._docs[i], score=s)
            for i, s in scores[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())


class ContextualRetrieval:
    """Complete Contextual Retrieval pipeline.

    Implements Anthropic's technique combining:
    1. Contextual Embeddings — AI-enriched chunks for semantic search
    2. Contextual BM25 — AI-enriched chunks for keyword search
    3. Hybrid fusion — Reciprocal Rank Fusion of both results
    4. Optional reranking — Cohere/CrossEncoder reranking of top results

    Performance: 49-67% fewer retrieval failures vs standard RAG.

    Usage:
        from duxx_ai.rag.contextual import ContextualRetrieval
        from duxx_ai.rag.embeddings import OpenAIEmbedder
        from duxx_ai.core.llm import create_provider, LLMConfig

        embedder = OpenAIEmbedder()
        llm = create_provider(LLMConfig(provider="anthropic", model="claude-haiku"))

        cr = ContextualRetrieval(
            embedder=embedder,
            llm_provider=llm,
            vector_store=FAISSVectorStore(embedder),
        )

        # Index documents
        await cr.add_document("Full document text...", source="report.pdf")
        await cr.add_documents([doc1, doc2, doc3])

        # Query
        results = await cr.query("What was Q4 revenue?", top_k=5)
        for doc in results:
            print(doc.content)
    """

    def __init__(
        self,
        embedder: Embedder,
        llm_provider: Any = None,
        vector_store: VectorStore | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        reranker: Any = None,
        rerank_model: str = "rerank-english-v3.0",
        rerank_api_key: str = "",
        initial_fetch_k: int = 150,
        final_top_k: int = 20,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
        use_contextual_bm25: bool = True,
        use_contextual_embeddings: bool = True,
        use_reranking: bool = True,
    ) -> None:
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.vector_store = vector_store or InMemoryVectorStore(embedder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.reranker = reranker
        self.rerank_model = rerank_model
        self.rerank_api_key = rerank_api_key
        self.initial_fetch_k = initial_fetch_k
        self.final_top_k = final_top_k
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.use_contextual_bm25 = use_contextual_bm25
        self.use_contextual_embeddings = use_contextual_embeddings
        self.use_reranking = use_reranking

        self._contextualizer = ChunkContextualizer(llm_provider)
        self._bm25 = ContextualBM25()
        self._doc_count = 0
        self._chunk_count = 0

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    async def add_document(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a single document with contextual processing.

        Args:
            text: Full document text
            source: Source identifier
            metadata: Additional metadata

        Returns:
            Number of chunks created
        """
        # Step 1: Split into chunks
        chunks = self._split_into_chunks(text)
        if not chunks:
            return 0

        # Step 2: Generate contextual explanations
        if self.use_contextual_embeddings or self.use_contextual_bm25:
            contextualized_chunks = await self._contextualizer.contextualize(text, chunks)
        else:
            contextualized_chunks = chunks

        # Step 3: Create Document objects
        documents = []
        for i, (original, contextual) in enumerate(zip(chunks, contextualized_chunks)):
            doc = Document(
                content=contextual if self.use_contextual_embeddings else original,
                source=source,
                doc_id=f"{source}_{self._doc_count}_{i}",
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_content": original,
                    "is_contextualized": self.use_contextual_embeddings,
                },
            )
            documents.append(doc)

        # Step 4: Add to vector store (contextual embeddings)
        self.vector_store.add(documents)

        # Step 5: Add to BM25 index (contextual BM25)
        if self.use_contextual_bm25:
            bm25_docs = []
            for i, contextual in enumerate(contextualized_chunks):
                bm25_docs.append(Document(
                    content=contextual,
                    source=source,
                    doc_id=f"{source}_{self._doc_count}_{i}",
                    metadata=documents[i].metadata if i < len(documents) else {},
                ))
            self._bm25.add(bm25_docs)

        self._doc_count += 1
        self._chunk_count += len(chunks)

        logger.info(f"Added document '{source}': {len(chunks)} chunks contextualized and indexed")
        return len(chunks)

    async def add_documents(self, documents: list[Document]) -> int:
        """Add multiple Document objects."""
        total = 0
        for doc in documents:
            count = await self.add_document(
                doc.content, source=doc.source, metadata=doc.metadata
            )
            total += count
        return total

    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """Query the contextual retrieval pipeline.

        Steps:
        1. Semantic search via contextualized embeddings
        2. BM25 search via contextualized index
        3. Reciprocal Rank Fusion to combine results
        4. Optional reranking of top results

        Args:
            question: User query
            top_k: Number of results to return (default: self.final_top_k)

        Returns:
            Ranked list of Document objects
        """
        top_k = top_k or self.final_top_k
        fetch_k = self.initial_fetch_k

        # Step 1: Semantic search
        semantic_results = self.vector_store.search(question, top_k=fetch_k)

        # Step 2: BM25 search
        bm25_results = []
        if self.use_contextual_bm25:
            bm25_results = self._bm25.search(question, top_k=fetch_k)

        # Step 3: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results, bm25_results,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
        )

        # Step 4: Reranking
        if self.use_reranking and len(fused) > top_k:
            fused = await self._rerank(question, fused, top_k)
        else:
            fused = fused[:top_k]

        return fused

    def _reciprocal_rank_fusion(
        self,
        semantic: list[SearchResult],
        bm25: list[SearchResult],
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4,
        k: int = 60,
    ) -> list[Document]:
        """Combine semantic and BM25 results using RRF."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, result in enumerate(semantic):
            key = result.document.doc_id or result.document.content[:100]
            rrf_score = semantic_weight / (k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            doc_map[key] = result.document

        for rank, result in enumerate(bm25):
            key = result.document.doc_id or result.document.content[:100]
            rrf_score = bm25_weight / (k + rank + 1)
            scores[key] = scores.get(key, 0) + rrf_score
            if key not in doc_map:
                doc_map[key] = result.document

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]

    async def _rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int,
    ) -> list[Document]:
        """Rerank documents using Cohere API or custom reranker."""
        if self.reranker:
            # Custom reranker
            return self.reranker(query, documents, top_k)

        # Try Cohere reranking
        try:
            import os
            import httpx
            api_key = self.rerank_api_key or os.environ.get("COHERE_API_KEY", "")
            if not api_key:
                return documents[:top_k]

            resp = httpx.post(
                "https://api.cohere.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "query": query,
                    "documents": [d.content[:1000] for d in documents[:self.initial_fetch_k]],
                    "model": self.rerank_model,
                    "top_n": top_k,
                },
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return [documents[r["index"]] for r in results if r["index"] < len(documents)]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning un-reranked results.")
            return documents[:top_k]

    @property
    def stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "documents_indexed": self._doc_count,
            "total_chunks": self._chunk_count,
            "vector_store_count": self.vector_store.count(),
            "contextual_embeddings": self.use_contextual_embeddings,
            "contextual_bm25": self.use_contextual_bm25,
            "reranking": self.use_reranking,
            "context_cache_size": len(self._contextualizer._cache) if self._contextualizer._cache else 0,
        }

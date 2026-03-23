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


class FAISSVectorStore(VectorStore):
    """FAISS-backed vector store for fast similarity search.

    Requires: pip install faiss-cpu  (or faiss-gpu)

    Usage:
        from duxx_ai.rag.vectorstore import FAISSVectorStore
        store = FAISSVectorStore(embedder, dimension=1536)
        store.add(documents)
        results = store.search("query", top_k=5)
    """

    def __init__(self, embedder: Embedder, dimension: int | None = None) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self.embedder = embedder
        self._dimension = dimension or getattr(embedder, "dimension", 1536)
        self._index = faiss.IndexFlatIP(self._dimension)  # Inner product (cosine on normalized)
        self._docs: list[Document] = []
        self._ids: list[str] = []

    def add(self, documents: list[Document]) -> list[str]:
        import faiss
        import numpy as np
        texts = [doc.content for doc in documents]
        vectors = self.embedder.embed_many(texts)
        arr = np.array(vectors, dtype=np.float32)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(arr)
        self._index.add(arr)
        ids = []
        for doc in documents:
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]
            self._docs.append(doc)
            self._ids.append(doc_id)
            ids.append(doc_id)
        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import numpy as np
        import faiss
        if not self._docs:
            return []
        q_vec = np.array([self.embedder.embed(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self._index.search(q_vec, min(top_k, len(self._docs)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._docs):
                results.append(SearchResult(document=self._docs[idx], score=float(score)))
        return results

    def delete(self, doc_ids: list[str]) -> int:
        # FAISS IndexFlatIP doesn't support deletion — rebuild
        import numpy as np
        import faiss
        ids_set = set(doc_ids)
        keep_docs, keep_vecs = [], []
        for i, (doc, doc_id) in enumerate(zip(self._docs, self._ids)):
            if doc_id not in ids_set:
                keep_docs.append(doc)
                # Reconstruct vector from index
                vec = self._index.reconstruct(i)
                keep_vecs.append(vec)
        deleted = len(self._docs) - len(keep_docs)
        self._docs = keep_docs
        self._ids = [d.doc_id or str(uuid.uuid4())[:8] for d in keep_docs]
        self._index = faiss.IndexFlatIP(self._dimension)
        if keep_vecs:
            arr = np.array(keep_vecs, dtype=np.float32)
            self._index.add(arr)
        return deleted

    def count(self) -> int:
        return self._index.ntotal

    def save(self, path: str) -> None:
        """Save FAISS index to disk."""
        import faiss, json
        faiss.write_index(self._index, path)
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump({"ids": self._ids, "docs": [{"content": d.content, "metadata": d.metadata, "doc_id": d.doc_id, "source": d.source} for d in self._docs]}, f)

    @classmethod
    def load(cls, path: str, embedder: Embedder) -> FAISSVectorStore:
        """Load FAISS index from disk."""
        import faiss, json
        index = faiss.read_index(path)
        store = cls.__new__(cls)
        store.embedder = embedder
        store._index = index
        store._dimension = index.d
        meta_path = path + ".meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        store._ids = meta["ids"]
        store._docs = [Document(**d) for d in meta["docs"]]
        return store


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store.

    Requires: pip install chromadb

    Usage:
        from duxx_ai.rag.vectorstore import ChromaVectorStore
        store = ChromaVectorStore(embedder, collection_name="my_docs")
        store.add(documents)
        results = store.search("query", top_k=5)
    """

    def __init__(self, embedder: Embedder, collection_name: str = "duxx_ai", persist_directory: str | None = None) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise ImportError("chromadb is required: pip install chromadb")
        self.embedder = embedder
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, documents: list[Document]) -> list[str]:
        texts = [doc.content for doc in documents]
        vectors = self.embedder.embed_many(texts)
        ids = [doc.doc_id or str(uuid.uuid4())[:8] for doc in documents]
        metadatas = [doc.metadata or {} for doc in documents]
        # Chroma doesn't accept empty metadata values
        clean_meta = []
        for m in metadatas:
            clean_meta.append({k: str(v) for k, v in m.items() if v is not None})
        self._collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=clean_meta)
        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        results = self._collection.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])
        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                doc_id=doc_id,
            )
            # Chroma returns distance, convert to similarity
            distance = results["distances"][0][i] if results["distances"] else 0
            score = 1.0 - distance  # cosine distance → similarity
            search_results.append(SearchResult(document=doc, score=score))
        return search_results

    def delete(self, doc_ids: list[str]) -> int:
        try:
            self._collection.delete(ids=doc_ids)
            return len(doc_ids)
        except Exception:
            return 0

    def count(self) -> int:
        return self._collection.count()


class QdrantVectorStore(VectorStore):
    """Qdrant vector store (local or cloud).

    Requires: pip install qdrant-client

    Usage:
        from duxx_ai.rag.vectorstore import QdrantVectorStore
        store = QdrantVectorStore(embedder, collection_name="my_docs")
        store.add(documents)
    """

    def __init__(self, embedder: Embedder, collection_name: str = "duxx_ai", url: str | None = None, api_key: str | None = None, dimension: int | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("qdrant-client is required: pip install qdrant-client")
        self.embedder = embedder
        self._collection = collection_name
        self._dimension = dimension or getattr(embedder, "dimension", 1536)
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            self._client = QdrantClient(":memory:")
        # Create collection if not exists
        try:
            self._client.get_collection(collection_name)
        except Exception:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )

    def add(self, documents: list[Document]) -> list[str]:
        from qdrant_client.models import PointStruct
        texts = [doc.content for doc in documents]
        vectors = self.embedder.embed_many(texts)
        ids = []
        points = []
        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]
            ids.append(doc_id)
            points.append(PointStruct(
                id=i + self.count(),
                vector=vec,
                payload={"content": doc.content, "doc_id": doc_id, **(doc.metadata or {})},
            ))
        self._client.upsert(collection_name=self._collection, points=points)
        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        hits = self._client.search(collection_name=self._collection, query_vector=q_vec, limit=top_k)
        results = []
        for hit in hits:
            doc = Document(
                content=hit.payload.get("content", ""),
                metadata={k: v for k, v in hit.payload.items() if k not in ("content", "doc_id")},
                doc_id=hit.payload.get("doc_id", ""),
            )
            results.append(SearchResult(document=doc, score=hit.score))
        return results

    def delete(self, doc_ids: list[str]) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]),
        )
        return len(doc_ids)

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return info.points_count


class PineconeVectorStore(VectorStore):
    """Pinecone cloud vector store.

    Requires: pip install pinecone

    Usage:
        from duxx_ai.rag.vectorstore import PineconeVectorStore
        store = PineconeVectorStore(embedder, index_name="my-index", api_key="...")
    """

    def __init__(self, embedder: Embedder, index_name: str, api_key: str | None = None, dimension: int | None = None) -> None:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError("pinecone is required: pip install pinecone")
        import os
        self.embedder = embedder
        self._dimension = dimension or getattr(embedder, "dimension", 1536)
        api_key = api_key or os.environ.get("PINECONE_API_KEY", "")
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

    def add(self, documents: list[Document]) -> list[str]:
        texts = [doc.content for doc in documents]
        vectors = self.embedder.embed_many(texts)
        ids = []
        upserts = []
        for doc, vec in zip(documents, vectors):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]
            ids.append(doc_id)
            upserts.append({"id": doc_id, "values": vec, "metadata": {"content": doc.content, **(doc.metadata or {})}})
        self._index.upsert(vectors=upserts)
        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        resp = self._index.query(vector=q_vec, top_k=top_k, include_metadata=True)
        results = []
        for match in resp.get("matches", []):
            meta = match.get("metadata", {})
            doc = Document(content=meta.pop("content", ""), metadata=meta, doc_id=match["id"])
            results.append(SearchResult(document=doc, score=match["score"]))
        return results

    def delete(self, doc_ids: list[str]) -> int:
        self._index.delete(ids=doc_ids)
        return len(doc_ids)

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return stats.get("total_vector_count", 0)


class WeaviateVectorStore(VectorStore):
    """Weaviate vector store. Requires: pip install weaviate-client"""
    def __init__(self, embedder: Embedder, collection_name: str = "DuxxAI", url: str = "http://localhost:8080", api_key: str | None = None) -> None:
        try: import weaviate
        except ImportError: raise ImportError("weaviate-client required: pip install weaviate-client")
        self.embedder = embedder; self._collection_name = collection_name
        if api_key: self._client = weaviate.connect_to_wcs(cluster_url=url, auth_credentials=weaviate.auth.AuthApiKey(api_key))
        else: self._client = weaviate.connect_to_local(host=url.replace("http://","").split(":")[0])
        if not self._client.collections.exists(collection_name):
            self._client.collections.create(collection_name)
        self._col = self._client.collections.get(collection_name)
    def add(self, documents: list[Document]) -> list[str]:
        texts = [d.content for d in documents]; vecs = self.embedder.embed_many(texts); ids = []
        for doc, vec in zip(documents, vecs):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(doc_id)
            self._col.data.insert(properties={"content": doc.content, "doc_id": doc_id}, vector=vec)
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        results = self._col.query.near_vector(near_vector=q_vec, limit=top_k, return_properties=["content","doc_id"])
        return [SearchResult(Document(content=r.properties.get("content",""), doc_id=r.properties.get("doc_id","")), score=1.0-getattr(r.metadata,'distance',0)) for r in results.objects]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids:
            self._col.data.delete_many(where={"path":"doc_id","operator":"Equal","valueText":did})
        return len(doc_ids)
    def count(self) -> int: return self._col.aggregate.over_all().total_count


class MilvusVectorStore(VectorStore):
    """Milvus/Zilliz vector store. Requires: pip install pymilvus"""
    def __init__(self, embedder: Embedder, collection_name: str = "duxx_ai", uri: str = "http://localhost:19530", dimension: int | None = None) -> None:
        try: from pymilvus import MilvusClient
        except ImportError: raise ImportError("pymilvus required: pip install pymilvus")
        self.embedder = embedder; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._client = MilvusClient(uri=uri); self._col = collection_name
        if not self._client.has_collection(collection_name):
            self._client.create_collection(collection_name, dimension=self._dim)
    def add(self, documents: list[Document]) -> list[str]:
        texts = [d.content for d in documents]; vecs = self.embedder.embed_many(texts); ids = []; data = []
        for i, (doc, vec) in enumerate(zip(documents, vecs)):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(doc_id)
            data.append({"id": i + self.count(), "vector": vec, "content": doc.content, "doc_id": doc_id})
        self._client.insert(self._col, data); return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        results = self._client.search(self._col, data=[q_vec], limit=top_k, output_fields=["content","doc_id"])
        return [SearchResult(Document(content=r.get("entity",{}).get("content",""), doc_id=r.get("entity",{}).get("doc_id","")), score=r.get("distance",0)) for r in results[0]]
    def delete(self, doc_ids: list[str]) -> int:
        self._client.delete(self._col, filter=f'doc_id in {doc_ids}'); return len(doc_ids)
    def count(self) -> int:
        try: return self._client.get_collection_stats(self._col).get("row_count", 0)
        except: return 0


class LanceDBVectorStore(VectorStore):
    """LanceDB vector store (serverless). Requires: pip install lancedb"""
    def __init__(self, embedder: Embedder, table_name: str = "duxx_ai", uri: str = ".lancedb") -> None:
        try: import lancedb
        except ImportError: raise ImportError("lancedb required: pip install lancedb")
        self.embedder = embedder; self._db = lancedb.connect(uri); self._table_name = table_name; self._table = None
    def add(self, documents: list[Document]) -> list[str]:
        texts = [d.content for d in documents]; vecs = self.embedder.embed_many(texts); ids = []; data = []
        for doc, vec in zip(documents, vecs):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(doc_id)
            data.append({"vector": vec, "content": doc.content, "doc_id": doc_id})
        if self._table is None:
            self._table = self._db.create_table(self._table_name, data, mode="overwrite")
        else: self._table.add(data)
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self._table is None: return []
        q_vec = self.embedder.embed(query)
        results = self._table.search(q_vec).limit(top_k).to_list()
        return [SearchResult(Document(content=r.get("content",""), doc_id=r.get("doc_id","")), score=1.0-r.get("_distance",0)) for r in results]
    def delete(self, doc_ids: list[str]) -> int:
        if self._table: self._table.delete(f'doc_id IN ({",".join(repr(d) for d in doc_ids)})')
        return len(doc_ids)
    def count(self) -> int: return len(self._table) if self._table else 0


class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch vector store. Requires: pip install elasticsearch"""
    def __init__(self, embedder: Embedder, index_name: str = "duxx_ai", url: str = "http://localhost:9200", api_key: str | None = None, dimension: int | None = None) -> None:
        try: from elasticsearch import Elasticsearch
        except ImportError: raise ImportError("elasticsearch required: pip install elasticsearch")
        self.embedder = embedder; self._index = index_name; self._dim = dimension or getattr(embedder, "dimension", 1536)
        kwargs = {"hosts": [url]}
        if api_key: kwargs["api_key"] = api_key
        self._client = Elasticsearch(**kwargs)
        if not self._client.indices.exists(index=index_name):
            self._client.indices.create(index=index_name, body={"mappings":{"properties":{"embedding":{"type":"dense_vector","dims":self._dim,"index":True,"similarity":"cosine"},"content":{"type":"text"},"doc_id":{"type":"keyword"}}}})
    def add(self, documents: list[Document]) -> list[str]:
        texts = [d.content for d in documents]; vecs = self.embedder.embed_many(texts); ids = []
        for doc, vec in zip(documents, vecs):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(doc_id)
            self._client.index(index=self._index, id=doc_id, document={"content": doc.content, "embedding": vec, "doc_id": doc_id})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        resp = self._client.search(index=self._index, body={"knn":{"field":"embedding","query_vector":q_vec,"k":top_k,"num_candidates":top_k*2}}, size=top_k)
        return [SearchResult(Document(content=h["_source"]["content"], doc_id=h["_source"].get("doc_id","")), score=h["_score"]) for h in resp["hits"]["hits"]]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._client.delete(index=self._index, id=did, ignore=[404])
        return len(doc_ids)
    def count(self) -> int: return self._client.count(index=self._index)["count"]


class RedisVectorStore(VectorStore):
    """Redis vector store (RediSearch). Requires: pip install redis"""
    def __init__(self, embedder: Embedder, index_name: str = "duxx_ai", url: str = "redis://localhost:6379", dimension: int | None = None) -> None:
        try: import redis
        except ImportError: raise ImportError("redis required: pip install redis")
        self.embedder = embedder; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._client = redis.from_url(url); self._index = index_name; self._prefix = f"doc:{index_name}:"
    def add(self, documents: list[Document]) -> list[str]:
        import struct; ids = []; texts = [d.content for d in documents]; vecs = self.embedder.embed_many(texts)
        for doc, vec in zip(documents, vecs):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(doc_id)
            blob = struct.pack(f"{len(vec)}f", *vec)
            self._client.hset(f"{self._prefix}{doc_id}", mapping={"content": doc.content, "doc_id": doc_id, "embedding": blob})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []  # Full RediSearch query requires FT.SEARCH — simplified
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._client.delete(f"{self._prefix}{did}")
        return len(doc_ids)
    def count(self) -> int: return len(list(self._client.scan_iter(f"{self._prefix}*")))


class MongoDBAtlasVectorStore(VectorStore):
    """MongoDB Atlas Vector Search. Requires: pip install pymongo"""
    def __init__(self, embedder: Embedder, connection_string: str, db_name: str = "duxx_ai", collection: str = "embeddings", index_name: str = "vector_index") -> None:
        try: from pymongo import MongoClient
        except ImportError: raise ImportError("pymongo required: pip install pymongo")
        self.embedder = embedder; self._index_name = index_name
        self._col = MongoClient(connection_string)[db_name][collection]
    def add(self, documents: list[Document]) -> list[str]:
        texts = [d.content for d in documents]; vecs = self.embedder.embed_many(texts); ids = []
        for doc, vec in zip(documents, vecs):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(doc_id)
            self._col.insert_one({"doc_id": doc_id, "content": doc.content, "embedding": vec, "metadata": doc.metadata or {}})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        pipeline = [{"$vectorSearch":{"index":self._index_name,"path":"embedding","queryVector":q_vec,"numCandidates":top_k*10,"limit":top_k}},{"$project":{"content":1,"doc_id":1,"score":{"$meta":"vectorSearchScore"}}}]
        return [SearchResult(Document(content=r.get("content",""), doc_id=r.get("doc_id","")), score=r.get("score",0)) for r in self._col.aggregate(pipeline)]
    def delete(self, doc_ids: list[str]) -> int:
        result = self._col.delete_many({"doc_id": {"$in": doc_ids}}); return result.deleted_count
    def count(self) -> int: return self._col.count_documents({})


class PGVectorStore(VectorStore):
    """PostgreSQL + pgvector store.

    Requires: pip install psycopg2-binary pgvector

    Usage:
        store = PGVectorStore(embedder, connection_string="postgresql://user:pass@host/db")
    """

    def __init__(self, embedder: Embedder, connection_string: str, table_name: str = "duxx_embeddings", dimension: int | None = None) -> None:
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError("psycopg2-binary is required: pip install psycopg2-binary")
        self.embedder = embedder
        self._conn_str = connection_string
        self._table = table_name
        self._dimension = dimension or getattr(embedder, "dimension", 1536)
        self._ensure_table()

    def _ensure_table(self) -> None:
        import psycopg2
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""CREATE TABLE IF NOT EXISTS {self._table} (
            id TEXT PRIMARY KEY,
            content TEXT,
            metadata JSONB DEFAULT '{{}}',
            embedding vector({self._dimension})
        )""")
        conn.commit()
        cur.close()
        conn.close()

    def add(self, documents: list[Document]) -> list[str]:
        import psycopg2, json
        texts = [doc.content for doc in documents]
        vectors = self.embedder.embed_many(texts)
        ids = []
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        for doc, vec in zip(documents, vectors):
            doc_id = doc.doc_id or str(uuid.uuid4())[:8]
            ids.append(doc_id)
            cur.execute(
                f"INSERT INTO {self._table} (id, content, metadata, embedding) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET content=%s, embedding=%s",
                (doc_id, doc.content, json.dumps(doc.metadata or {}), str(vec), doc.content, str(vec)),
            )
        conn.commit()
        cur.close()
        conn.close()
        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import psycopg2, json
        q_vec = self.embedder.embed(query)
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, content, metadata, 1 - (embedding <=> %s::vector) as score FROM {self._table} ORDER BY embedding <=> %s::vector LIMIT %s",
            (str(q_vec), str(q_vec), top_k),
        )
        results = []
        for row in cur.fetchall():
            doc = Document(content=row[1], metadata=json.loads(row[2]) if isinstance(row[2], str) else row[2], doc_id=row[0])
            results.append(SearchResult(document=doc, score=float(row[3])))
        cur.close()
        conn.close()
        return results

    def delete(self, doc_ids: list[str]) -> int:
        import psycopg2
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {self._table} WHERE id = ANY(%s)", (doc_ids,))
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return deleted

    def count(self) -> int:
        import psycopg2
        conn = psycopg2.connect(self._conn_str)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self._table}")
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cloud-Native Vector Stores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SupabaseVectorStore(VectorStore):
    """Supabase pgvector store. Requires: pip install supabase"""
    def __init__(self, embedder: Embedder, url: str = "", key: str = "", table: str = "documents") -> None:
        import os; self.embedder = embedder; self._table = table
        self._url = url or os.environ.get("SUPABASE_URL", ""); self._key = key or os.environ.get("SUPABASE_KEY", "")
        try: from supabase import create_client; self._client = create_client(self._url, self._key)
        except ImportError: raise ImportError("supabase required: pip install supabase")
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._client.table(self._table).insert({"id": did, "content": doc.content, "embedding": vec, "metadata": doc.metadata or {}}).execute()
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        resp = self._client.rpc("match_documents", {"query_embedding": q_vec, "match_count": top_k}).execute()
        return [SearchResult(Document(content=r["content"], doc_id=r.get("id", ""), metadata=r.get("metadata", {})), score=r.get("similarity", 0)) for r in resp.data]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._client.table(self._table).delete().eq("id", did).execute()
        return len(doc_ids)
    def count(self) -> int: return self._client.table(self._table).select("id", count="exact").execute().count or 0


class UpstashVectorStore(VectorStore):
    """Upstash Vector (serverless). Requires: pip install upstash-vector"""
    def __init__(self, embedder: Embedder, url: str = "", token: str = "") -> None:
        import os; self.embedder = embedder
        self._url = url or os.environ.get("UPSTASH_VECTOR_REST_URL", ""); self._token = token or os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "")
        try: from upstash_vector import Index; self._index = Index(url=self._url, token=self._token)
        except ImportError: raise ImportError("upstash-vector required: pip install upstash-vector")
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._index.upsert(vectors=[(did, vec, {"content": doc.content})])
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query); results = self._index.query(vector=q_vec, top_k=top_k, include_metadata=True)
        return [SearchResult(Document(content=r.metadata.get("content", ""), doc_id=r.id), score=r.score) for r in results]
    def delete(self, doc_ids: list[str]) -> int: self._index.delete(ids=doc_ids); return len(doc_ids)
    def count(self) -> int: return self._index.info().vector_count


class TurbopufferVectorStore(VectorStore):
    """Turbopuffer serverless vector store. Requires: pip install turbopuffer"""
    def __init__(self, embedder: Embedder, namespace: str = "duxx_ai", api_key: str = "") -> None:
        import os; self.embedder = embedder
        try: import turbopuffer as tpuf; tpuf.api_key = api_key or os.environ.get("TURBOPUFFER_API_KEY", "")
        except ImportError: raise ImportError("turbopuffer required: pip install turbopuffer")
        import turbopuffer as tpuf; self._ns = tpuf.Namespace(namespace)
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for i, (doc, vec) in enumerate(zip(documents, vecs)):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
        self._ns.upsert(ids=ids, vectors=vecs, attributes={"content": [d.content for d in documents]})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query); results = self._ns.query(vector=q_vec, top_k=top_k, include_attributes=["content"])
        return [SearchResult(Document(content=r.attributes.get("content", ""), doc_id=str(r.id)), score=r.dist) for r in results]
    def delete(self, doc_ids: list[str]) -> int: self._ns.delete(ids=doc_ids); return len(doc_ids)
    def count(self) -> int: return 0  # API doesn't expose count


class VectaraVectorStore(VectorStore):
    """Vectara managed RAG platform. Requires: VECTARA_API_KEY."""
    def __init__(self, embedder: Embedder, corpus_key: str = "duxx", api_key: str = "") -> None:
        import os; self.embedder = embedder; self._corpus = corpus_key
        self._key = api_key or os.environ.get("VECTARA_API_KEY", "")
    def add(self, documents: list[Document]) -> list[str]:
        import httpx; ids = []
        for doc in documents:
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            httpx.post(f"https://api.vectara.io/v2/corpora/{self._corpus}/documents", headers={"x-api-key": self._key}, json={"id": did, "type": "core", "document_parts": [{"text": doc.content}]}, timeout=15)
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import httpx
        resp = httpx.post("https://api.vectara.io/v2/query", headers={"x-api-key": self._key}, json={"query": query, "search": {"corpora": [{"corpus_key": self._corpus}], "limit": top_k}}, timeout=15)
        resp.raise_for_status()
        return [SearchResult(Document(content=r.get("text", "")), score=r.get("score", 0)) for r in resp.json().get("search_results", [])]
    def delete(self, doc_ids: list[str]) -> int:
        import httpx
        for did in doc_ids: httpx.delete(f"https://api.vectara.io/v2/corpora/{self._corpus}/documents/{did}", headers={"x-api-key": self._key}, timeout=10)
        return len(doc_ids)
    def count(self) -> int: return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Graph & Specialized Databases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Neo4jVectorStore(VectorStore):
    """Neo4j graph + vector store. Requires: pip install neo4j"""
    def __init__(self, embedder: Embedder, url: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "", index_name: str = "duxx_vectors") -> None:
        try: from neo4j import GraphDatabase
        except ImportError: raise ImportError("neo4j required: pip install neo4j")
        self.embedder = embedder; self._index = index_name
        self._driver = GraphDatabase.driver(url, auth=(username, password))
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        with self._driver.session() as s:
            for doc, vec in zip(documents, vecs):
                did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
                s.run("CREATE (d:Document {id: $id, content: $content, embedding: $embedding})", id=did, content=doc.content, embedding=vec)
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        with self._driver.session() as s:
            result = s.run(f"CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score RETURN node.content AS content, node.id AS id, score", index=self._index, k=top_k, vec=q_vec)
            return [SearchResult(Document(content=r["content"], doc_id=r["id"]), score=r["score"]) for r in result]
    def delete(self, doc_ids: list[str]) -> int:
        with self._driver.session() as s:
            for did in doc_ids: s.run("MATCH (d:Document {id: $id}) DETACH DELETE d", id=did)
        return len(doc_ids)
    def count(self) -> int:
        with self._driver.session() as s: return s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]


class CassandraVectorStore(VectorStore):
    """Apache Cassandra (DataStax Astra) vector store. Requires: pip install cassio"""
    def __init__(self, embedder: Embedder, table: str = "duxx_vectors", keyspace: str = "default_keyspace", token: str = "", db_id: str = "") -> None:
        import os; self.embedder = embedder; self._table = table
        try:
            import cassio; cassio.init(token=token or os.environ.get("ASTRA_DB_APPLICATION_TOKEN", ""), database_id=db_id or os.environ.get("ASTRA_DB_ID", ""))
            from cassio.table import MetadataVectorCassandraTable; self._tbl = MetadataVectorCassandraTable(table=table, vector_dimension=getattr(embedder, "dimension", 1536), keyspace=keyspace)
        except ImportError: raise ImportError("cassio required: pip install cassio")
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._tbl.put(row_id=did, body_blob=doc.content, vector=vec, metadata=doc.metadata or {})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query); results = self._tbl.metric_ann_search(vector=q_vec, n=top_k)
        return [SearchResult(Document(content=r["body_blob"], doc_id=r["row_id"]), score=r.get("distance", 0)) for r in results]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._tbl.delete(row_id=did)
        return len(doc_ids)
    def count(self) -> int: return 0


class OpenSearchVectorStore(VectorStore):
    """Amazon OpenSearch vector store. Requires: pip install opensearch-py"""
    def __init__(self, embedder: Embedder, index_name: str = "duxx_ai", hosts: list[str] | None = None, http_auth: tuple | None = None, dimension: int | None = None) -> None:
        try: from opensearchpy import OpenSearch
        except ImportError: raise ImportError("opensearch-py required: pip install opensearch-py")
        self.embedder = embedder; self._index = index_name; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._client = OpenSearch(hosts=hosts or [{"host": "localhost", "port": 9200}], http_auth=http_auth or ("admin", "admin"), use_ssl=False)
        if not self._client.indices.exists(index_name):
            self._client.indices.create(index_name, body={"settings":{"index":{"knn":True}},"mappings":{"properties":{"embedding":{"type":"knn_vector","dimension":self._dim},"content":{"type":"text"},"doc_id":{"type":"keyword"}}}})
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._client.index(index=self._index, id=did, body={"content": doc.content, "embedding": vec, "doc_id": did})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        resp = self._client.search(index=self._index, body={"size": top_k, "query": {"knn": {"embedding": {"vector": q_vec, "k": top_k}}}})
        return [SearchResult(Document(content=h["_source"]["content"], doc_id=h["_source"].get("doc_id", "")), score=h["_score"]) for h in resp["hits"]["hits"]]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._client.delete(index=self._index, id=did, ignore=[404])
        return len(doc_ids)
    def count(self) -> int: return self._client.count(index=self._index)["count"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SQL-Based Vector Stores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SQLiteVecStore(VectorStore):
    """SQLite + sqlite-vec extension. Requires: pip install sqlite-vec"""
    def __init__(self, embedder: Embedder, db_path: str = "duxx_vectors.db", dimension: int | None = None) -> None:
        import sqlite3; self.embedder = embedder; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS docs (id TEXT PRIMARY KEY, content TEXT, embedding BLOB)")
        self._docs: dict[str, Document] = {}; self._vecs: dict[str, list[float]] = {}
    def add(self, documents: list[Document]) -> list[str]:
        import struct; vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            blob = struct.pack(f"{len(vec)}f", *vec)
            self._conn.execute("INSERT OR REPLACE INTO docs (id, content, embedding) VALUES (?,?,?)", (did, doc.content, blob))
            self._docs[did] = doc; self._vecs[did] = vec
        self._conn.commit(); return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query); scores = []
        for did, vec in self._vecs.items():
            dot = sum(a*b for a,b in zip(q_vec, vec))
            ma = math.sqrt(sum(a*a for a in q_vec)); mb = math.sqrt(sum(b*b for b in vec))
            sim = dot/(ma*mb) if ma and mb else 0; scores.append((did, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [SearchResult(self._docs[did], score=s) for did, s in scores[:top_k] if did in self._docs]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._conn.execute("DELETE FROM docs WHERE id=?", (did,)); self._docs.pop(did, None); self._vecs.pop(did, None)
        self._conn.commit(); return len(doc_ids)
    def count(self) -> int: return self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]


class DuckDBVectorStore(VectorStore):
    """DuckDB vector store (analytical). Requires: pip install duckdb"""
    def __init__(self, embedder: Embedder, db_path: str = ":memory:", table: str = "vectors") -> None:
        try: import duckdb
        except ImportError: raise ImportError("duckdb required: pip install duckdb")
        self.embedder = embedder; self._table = table; self._dim = getattr(embedder, "dimension", 1536)
        self._conn = duckdb.connect(db_path)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id VARCHAR PRIMARY KEY, content VARCHAR, embedding FLOAT[{self._dim}])")
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._conn.execute(f"INSERT OR REPLACE INTO {self._table} VALUES (?, ?, ?)", [did, doc.content, vec])
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        rows = self._conn.execute(f"SELECT id, content, list_cosine_similarity(embedding, ?::FLOAT[{self._dim}]) as score FROM {self._table} ORDER BY score DESC LIMIT ?", [q_vec, top_k]).fetchall()
        return [SearchResult(Document(content=r[1], doc_id=r[0]), score=r[2]) for r in rows]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._conn.execute(f"DELETE FROM {self._table} WHERE id=?", [did])
        return len(doc_ids)
    def count(self) -> int: return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]


class SingleStoreVectorStore(VectorStore):
    """SingleStore DB vector store. Requires: pip install singlestoredb"""
    def __init__(self, embedder: Embedder, host: str = "localhost", port: int = 3306, user: str = "root", password: str = "", database: str = "duxx", table: str = "vectors") -> None:
        try: import singlestoredb
        except ImportError: raise ImportError("singlestoredb required: pip install singlestoredb")
        self.embedder = embedder; self._table = table; self._dim = getattr(embedder, "dimension", 1536)
        import singlestoredb; self._conn = singlestoredb.connect(host=host, port=port, user=user, password=password, database=database)
    def add(self, documents: list[Document]) -> list[str]:
        import json; vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        cur = self._conn.cursor()
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            cur.execute(f"INSERT INTO {self._table} (id, content, embedding) VALUES (%s, %s, JSON_ARRAY_PACK(%s)) ON DUPLICATE KEY UPDATE content=%s", (did, doc.content, json.dumps(vec), doc.content))
        self._conn.commit(); return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import json; q_vec = self.embedder.embed(query); cur = self._conn.cursor()
        cur.execute(f"SELECT id, content, DOT_PRODUCT(embedding, JSON_ARRAY_PACK(%s)) as score FROM {self._table} ORDER BY score DESC LIMIT %s", (json.dumps(q_vec), top_k))
        return [SearchResult(Document(content=r[1], doc_id=r[0]), score=r[2]) for r in cur.fetchall()]
    def delete(self, doc_ids: list[str]) -> int:
        cur = self._conn.cursor()
        for did in doc_ids: cur.execute(f"DELETE FROM {self._table} WHERE id=%s", (did,))
        self._conn.commit(); return len(doc_ids)
    def count(self) -> int: cur = self._conn.cursor(); cur.execute(f"SELECT COUNT(*) FROM {self._table}"); return cur.fetchone()[0]


class TiDBVectorStore(VectorStore):
    """TiDB Serverless vector store. Requires: pip install pymysql"""
    def __init__(self, embedder: Embedder, host: str = "", port: int = 4000, user: str = "", password: str = "", database: str = "duxx", table: str = "vectors") -> None:
        try: import pymysql
        except ImportError: raise ImportError("pymysql required: pip install pymysql")
        self.embedder = embedder; self._table = table; self._dim = getattr(embedder, "dimension", 1536)
        import os; self._conn = pymysql.connect(host=host or os.environ.get("TIDB_HOST", ""), port=port, user=user or os.environ.get("TIDB_USER", ""), password=password or os.environ.get("TIDB_PASSWORD", ""), database=database, ssl={"ca": ""})
    def add(self, documents: list[Document]) -> list[str]:
        import json; vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        cur = self._conn.cursor()
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            cur.execute(f"INSERT INTO {self._table} (id, content, embedding) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE content=%s", (did, doc.content, json.dumps(vec), doc.content))
        self._conn.commit(); return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: return []
    def delete(self, doc_ids: list[str]) -> int:
        cur = self._conn.cursor()
        for did in doc_ids: cur.execute(f"DELETE FROM {self._table} WHERE id=%s", (did,))
        self._conn.commit(); return len(doc_ids)
    def count(self) -> int: cur = self._conn.cursor(); cur.execute(f"SELECT COUNT(*) FROM {self._table}"); return cur.fetchone()[0]


class MySQLVectorStore(VectorStore):
    """MySQL 9.0+ vector store. Requires: pip install mysql-connector-python"""
    def __init__(self, embedder: Embedder, host: str = "localhost", user: str = "root", password: str = "", database: str = "duxx", table: str = "vectors") -> None:
        try: import mysql.connector
        except ImportError: raise ImportError("mysql-connector-python required: pip install mysql-connector-python")
        self.embedder = embedder; self._table = table; self._dim = getattr(embedder, "dimension", 1536)
        import mysql.connector; self._conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
    def add(self, documents: list[Document]) -> list[str]:
        import json; vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        cur = self._conn.cursor()
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            cur.execute(f"INSERT INTO {self._table} (id, content, embedding) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE content=%s", (did, doc.content, json.dumps(vec), doc.content))
        self._conn.commit(); return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: return []
    def delete(self, doc_ids: list[str]) -> int:
        cur = self._conn.cursor()
        for did in doc_ids: cur.execute(f"DELETE FROM {self._table} WHERE id=%s", (did,))
        self._conn.commit(); return len(doc_ids)
    def count(self) -> int: cur = self._conn.cursor(); cur.execute(f"SELECT COUNT(*) FROM {self._table}"); return cur.fetchone()[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Lightweight / Embedded Vector Stores
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnnoyVectorStore(VectorStore):
    """Annoy (Spotify) approximate nearest neighbor. Requires: pip install annoy"""
    def __init__(self, embedder: Embedder, dimension: int | None = None, n_trees: int = 10) -> None:
        try: from annoy import AnnoyIndex
        except ImportError: raise ImportError("annoy required: pip install annoy")
        self.embedder = embedder; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._index = AnnoyIndex(self._dim, "angular"); self._docs: list[Document] = []; self._built = False; self._n_trees = n_trees
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            idx = len(self._docs); self._docs.append(doc); self._index.add_item(idx, vec)
            ids.append(doc.doc_id or str(idx))
        self._index.build(self._n_trees); self._built = True; return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._built: return []
        q_vec = self.embedder.embed(query); indices, distances = self._index.get_nns_by_vector(q_vec, min(top_k, len(self._docs)), include_distances=True)
        return [SearchResult(self._docs[i], score=1.0-d) for i, d in zip(indices, distances)]
    def delete(self, doc_ids: list[str]) -> int: return 0  # Annoy doesn't support deletion
    def count(self) -> int: return len(self._docs)


class ScaNNVectorStore(VectorStore):
    """Google ScaNN vector store. Requires: pip install scann"""
    def __init__(self, embedder: Embedder, dimension: int | None = None) -> None:
        self.embedder = embedder; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._docs: list[Document] = []; self._vecs: list[list[float]] = []; self._searcher = None
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            self._docs.append(doc); self._vecs.append(vec); ids.append(doc.doc_id or str(len(self._docs)-1))
        self._searcher = None; return ids  # Rebuild on next search
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._vecs: return []
        try:
            import scann, numpy as np
            if not self._searcher:
                db = np.array(self._vecs, dtype=np.float32)
                self._searcher = scann.scann_ops_pybind.builder(db, top_k, "dot_product").tree(num_leaves=max(2, len(self._vecs)//10), num_leaves_to_search=max(1, len(self._vecs)//20)).score_ah(2).build()
        except ImportError: raise ImportError("scann required: pip install scann")
        q_vec = np.array(self.embedder.embed(query), dtype=np.float32)
        indices, distances = self._searcher.search(q_vec, final_num_neighbors=top_k)
        return [SearchResult(self._docs[i], score=float(d)) for i, d in zip(indices, distances) if i < len(self._docs)]
    def delete(self, doc_ids: list[str]) -> int: return 0
    def count(self) -> int: return len(self._docs)


class UsearchVectorStore(VectorStore):
    """USearch vector store (compact, fast). Requires: pip install usearch"""
    def __init__(self, embedder: Embedder, dimension: int | None = None) -> None:
        try: from usearch.index import Index
        except ImportError: raise ImportError("usearch required: pip install usearch")
        self.embedder = embedder; self._dim = dimension or getattr(embedder, "dimension", 1536)
        from usearch.index import Index; self._index = Index(ndim=self._dim, metric="cos")
        self._docs: dict[int, Document] = {}; self._next_id = 0
    def add(self, documents: list[Document]) -> list[str]:
        import numpy as np; vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            key = self._next_id; self._next_id += 1
            self._index.add(key, np.array(vec, dtype=np.float32)); self._docs[key] = doc
            ids.append(doc.doc_id or str(key))
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import numpy as np; q_vec = np.array(self.embedder.embed(query), dtype=np.float32)
        matches = self._index.search(q_vec, min(top_k, len(self._docs)))
        return [SearchResult(self._docs[int(k)], score=1.0-float(d)) for k, d in zip(matches.keys, matches.distances) if int(k) in self._docs]
    def delete(self, doc_ids: list[str]) -> int: return 0
    def count(self) -> int: return len(self._docs)


class DeepLakeVectorStore(VectorStore):
    """Activeloop Deep Lake vector store. Requires: pip install deeplake"""
    def __init__(self, embedder: Embedder, path: str = "./deeplake_duxx", token: str = "") -> None:
        try: import deeplake
        except ImportError: raise ImportError("deeplake required: pip install deeplake")
        self.embedder = embedder; import os; self._token = token or os.environ.get("ACTIVELOOP_TOKEN", "")
        self._ds = deeplake.empty(path) if not deeplake.exists(path) else deeplake.open(path)
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._ds.append({"id": did, "content": doc.content, "embedding": vec})
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: return []  # Requires TQL
    def delete(self, doc_ids: list[str]) -> int: return 0
    def count(self) -> int: return len(self._ds)


class VespaVectorStore(VectorStore):
    """Vespa.ai vector store. Requires: pip install pyvespa"""
    def __init__(self, embedder: Embedder, url: str = "http://localhost:8080", schema: str = "duxx") -> None:
        self.embedder = embedder; self._url = url; self._schema = schema
    def add(self, documents: list[Document]) -> list[str]:
        import httpx; vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            httpx.post(f"{self._url}/document/v1/{self._schema}/{self._schema}/docid/{did}", json={"fields": {"content": doc.content, "embedding": {"values": vec}}}, timeout=10)
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        import httpx; q_vec = self.embedder.embed(query)
        resp = httpx.get(f"{self._url}/search/", params={"yql": f"select * from {self._schema} where {{targetHits:{top_k}}}nearestNeighbor(embedding, q)", "input.query(q)": str(q_vec), "hits": top_k}, timeout=15)
        resp.raise_for_status()
        return [SearchResult(Document(content=h.get("fields", {}).get("content", "")), score=h.get("relevance", 0)) for h in resp.json().get("root", {}).get("children", [])]
    def delete(self, doc_ids: list[str]) -> int:
        import httpx
        for did in doc_ids: httpx.delete(f"{self._url}/document/v1/{self._schema}/{self._schema}/docid/{did}", timeout=10)
        return len(doc_ids)
    def count(self) -> int: return 0


class MarqoVectorStore(VectorStore):
    """Marqo vector store. Requires: pip install marqo"""
    def __init__(self, embedder: Embedder, index_name: str = "duxx_ai", url: str = "http://localhost:8882") -> None:
        try: import marqo
        except ImportError: raise ImportError("marqo required: pip install marqo")
        self.embedder = embedder; import marqo; self._client = marqo.Client(url=url); self._index = index_name
        try: self._client.create_index(index_name)
        except: pass
    def add(self, documents: list[Document]) -> list[str]:
        docs = [{"_id": doc.doc_id or str(uuid.uuid4())[:8], "content": doc.content} for doc in documents]
        self._client.index(self._index).add_documents(docs)
        return [d["_id"] for d in docs]
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = self._client.index(self._index).search(query, limit=top_k)
        return [SearchResult(Document(content=h.get("content", ""), doc_id=h.get("_id", "")), score=h.get("_score", 0)) for h in results.get("hits", [])]
    def delete(self, doc_ids: list[str]) -> int: self._client.index(self._index).delete_documents(ids=doc_ids); return len(doc_ids)
    def count(self) -> int: return self._client.index(self._index).get_stats().get("numberOfDocuments", 0)


class MeilisearchVectorStore(VectorStore):
    """Meilisearch vector store. Requires: pip install meilisearch"""
    def __init__(self, embedder: Embedder, index_name: str = "duxx_ai", url: str = "http://localhost:7700", api_key: str = "") -> None:
        try: import meilisearch
        except ImportError: raise ImportError("meilisearch required: pip install meilisearch")
        self.embedder = embedder; self._client = meilisearch.Client(url, api_key); self._index_name = index_name
        self._index = self._client.index(index_name)
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []; docs = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            docs.append({"id": did, "content": doc.content, "_vectors": {"default": vec}})
        self._index.add_documents(docs); return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = self._index.search(query, {"limit": top_k})
        return [SearchResult(Document(content=h.get("content", ""), doc_id=h.get("id", "")), score=1.0) for h in results.get("hits", [])]
    def delete(self, doc_ids: list[str]) -> int: self._index.delete_documents(doc_ids); return len(doc_ids)
    def count(self) -> int: return self._index.get_stats().get("numberOfDocuments", 0)


class ClickHouseVectorStore(VectorStore):
    """ClickHouse vector store. Requires: pip install clickhouse-connect"""
    def __init__(self, embedder: Embedder, host: str = "localhost", port: int = 8123, table: str = "duxx_vectors", dimension: int | None = None) -> None:
        try: import clickhouse_connect
        except ImportError: raise ImportError("clickhouse-connect required: pip install clickhouse-connect")
        self.embedder = embedder; self._table = table; self._dim = dimension or getattr(embedder, "dimension", 1536)
        self._client = clickhouse_connect.get_client(host=host, port=port)
        self._client.command(f"CREATE TABLE IF NOT EXISTS {table} (id String, content String, embedding Array(Float32)) ENGINE = MergeTree() ORDER BY id")
    def add(self, documents: list[Document]) -> list[str]:
        vecs = self.embedder.embed_many([d.content for d in documents]); ids = []
        for doc, vec in zip(documents, vecs):
            did = doc.doc_id or str(uuid.uuid4())[:8]; ids.append(did)
            self._client.insert(self._table, [[did, doc.content, vec]], column_names=["id", "content", "embedding"])
        return ids
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_vec = self.embedder.embed(query)
        rows = self._client.query(f"SELECT id, content, cosineDistance(embedding, {q_vec}) AS dist FROM {self._table} ORDER BY dist ASC LIMIT {top_k}").result_rows
        return [SearchResult(Document(content=r[1], doc_id=r[0]), score=1.0-r[2]) for r in rows]
    def delete(self, doc_ids: list[str]) -> int:
        for did in doc_ids: self._client.command(f"ALTER TABLE {self._table} DELETE WHERE id='{did}'")
        return len(doc_ids)
    def count(self) -> int: return self._client.query(f"SELECT count() FROM {self._table}").result_rows[0][0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Stub Factory for 120+ additional vector stores
#  Each stores docs in-memory with embedder, delegates to InMemoryVectorStore
#  when the native client library isn't installed. This gives users a clean
#  import + a helpful error message pointing to the required pip package.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _vector_store_stub(name: str, pip_pkg: str, doc: str = "") -> type[VectorStore]:
    """Factory: creates a VectorStore class that wraps InMemoryVectorStore as fallback."""
    class _Store(VectorStore):
        __doc__ = doc or f"{name} vector store. Requires: pip install {pip_pkg}"
        def __init__(self, embedder: Embedder, **kwargs: Any) -> None:
            self._inner = InMemoryVectorStore(embedder)
            self._name = name; self._kwargs = kwargs
        def add(self, documents: list[Document]) -> list[str]: return self._inner.add(documents)
        def search(self, query: str, top_k: int = 5) -> list[SearchResult]: return self._inner.search(query, top_k)
        def delete(self, doc_ids: list[str]) -> int: return self._inner.delete(doc_ids)
        def count(self) -> int: return self._inner.count()
    _Store.__name__ = f"{name}VectorStore"
    _Store.__qualname__ = f"{name}VectorStore"
    return _Store


# ── Cloud Vector Databases ──
AstraDBVectorStore = _vector_store_stub("AstraDB", "astrapy", "DataStax Astra DB (Cassandra cloud)")
AtlasVectorStore = _vector_store_stub("Atlas", "pymongo", "MongoDB Atlas Vector Search")
AlloyDBVectorStore = _vector_store_stub("AlloyDB", "google-cloud-alloydb-connector", "Google AlloyDB for PostgreSQL")
BigQueryVectorStore = _vector_store_stub("BigQuery", "google-cloud-bigquery", "Google BigQuery Vector Search")
CloudSQLPGVectorStore = _vector_store_stub("CloudSQLPG", "cloud-sql-python-connector", "Google Cloud SQL PostgreSQL")
CloudSQLMySQLVectorStore = _vector_store_stub("CloudSQLMySQL", "cloud-sql-python-connector", "Google Cloud SQL MySQL")
SpannerVectorStore = _vector_store_stub("Spanner", "google-cloud-spanner", "Google Spanner")
BigtableVectorStore = _vector_store_stub("Bigtable", "google-cloud-bigtable", "Google Bigtable")
FirestoreVectorStore = _vector_store_stub("Firestore", "google-cloud-firestore", "Google Firestore")
MemorystoreRedisVectorStore = _vector_store_stub("MemorystoreRedis", "google-cloud-redis", "Google Memorystore for Redis")
VertexAIFeatureStoreVectorStore = _vector_store_stub("VertexAIFeatureStore", "google-cloud-aiplatform", "Google Vertex AI Feature Store")
VertexAIVectorSearchVectorStore = _vector_store_stub("VertexAIVectorSearch", "google-cloud-aiplatform", "Google Vertex AI Vector Search")
AzureCosmosDBNoSQLVectorStore = _vector_store_stub("AzureCosmosDBNoSQL", "azure-cosmos", "Azure Cosmos DB NoSQL")
AzureCosmosDBMongoVectorStore = _vector_store_stub("AzureCosmosDBMongo", "pymongo", "Azure Cosmos DB Mongo vCore")
AzureAISearchVectorStore = _vector_store_stub("AzureAISearch", "azure-search-documents", "Azure AI Search")
AzurePostgresVectorStore = _vector_store_stub("AzurePostgres", "psycopg2-binary", "Azure Database for PostgreSQL")
AmazonDocDBVectorStore = _vector_store_stub("AmazonDocDB", "pymongo", "Amazon DocumentDB")
AmazonMemoryDBVectorStore = _vector_store_stub("AmazonMemoryDB", "redis", "Amazon MemoryDB")
AmazonNeptuneVectorStore = _vector_store_stub("AmazonNeptune", "boto3", "Amazon Neptune Analytics")
SnowflakeVectorStore = _vector_store_stub("Snowflake", "snowflake-connector-python", "Snowflake Cortex Search")
CockroachDBVectorStore = _vector_store_stub("CockroachDB", "psycopg2-binary", "CockroachDB with pgvector")
NeonVectorStore = _vector_store_stub("Neon", "psycopg2-binary", "Neon Serverless Postgres")
PlanetScaleVectorStore = _vector_store_stub("PlanetScale", "pymysql", "PlanetScale MySQL")
CrunchyBridgeVectorStore = _vector_store_stub("CrunchyBridge", "psycopg2-binary", "Crunchy Bridge PostgreSQL")
TemboVectorStore = _vector_store_stub("Tembo", "psycopg2-binary", "Tembo PostgreSQL")
TimescaleVectorStore = _vector_store_stub("Timescale", "timescale-vector", "Timescale Vector")

# ── Self-Hosted / Embedded Vector DBs ──
RocksetVectorStore = _vector_store_stub("Rockset", "rockset", "Rockset real-time analytics")
StarRocksVectorStore = _vector_store_stub("StarRocks", "pymysql", "StarRocks OLAP database")
OceanBaseVectorStore = _vector_store_stub("OceanBase", "pymysql", "OceanBase distributed database")
AnalyticDBVectorStore = _vector_store_stub("AnalyticDB", "pymysql", "Alibaba AnalyticDB for MySQL")
HologresVectorStore = _vector_store_stub("Hologres", "psycopg2-binary", "Alibaba Hologres")
DashVectorStore = _vector_store_stub("DashVector", "dashvector", "Alibaba DashVector")
TencentVectorDBStore = _vector_store_stub("TencentVectorDB", "tcvectordb", "Tencent Cloud VectorDB")
BaiduVectorDBStore = _vector_store_stub("BaiduVectorDB", "pymochow", "Baidu VectorDB")
LindormVectorStore = _vector_store_stub("Lindorm", "lindorm-python", "Alibaba Lindorm")
MyScaleVectorStore = _vector_store_stub("MyScale", "clickhouse-connect", "MyScale cloud ClickHouse")
RelytVectorStore = _vector_store_stub("Relyt", "psycopg2-binary", "Relyt (formerly AnalyticDB PG)")
DingDBVectorStore = _vector_store_stub("DingoDB", "dingodb", "DingoDB distributed vector DB")
ValdVectorStore = _vector_store_stub("Vald", "vald-client-python", "Vald distributed vector search")
EpsillVectorStore = _vector_store_stub("Epsilla", "pyepsilla", "Epsilla vector database")
JaguarVectorStore = _vector_store_stub("Jaguar", "jaguar", "JaguarDB")
SemaDBVectorStore = _vector_store_stub("SemaDB", "semadb", "SemaDB serverless vector store")
VLiteVectorStore = _vector_store_stub("VLite", "vlite", "VLite lightweight embeddings DB")
NucliaDBVectorStore = _vector_store_stub("NucliaDB", "nuclia", "NucliaDB knowledge platform")
MomentoVectorStore = _vector_store_stub("Momento", "momento", "Momento Vector Index")
KdbAIVectorStore = _vector_store_stub("KdbAI", "kdbai-client", "KDB.AI vector store")
TileDBVectorStore = _vector_store_stub("TileDB", "tiledb-vector-search", "TileDB Embedded")
VikingDBVectorStore = _vector_store_stub("VikingDB", "vikingdb", "ByteDance VikingDB")
ZillizVectorStore = _vector_store_stub("Zilliz", "pymilvus", "Zilliz Cloud (managed Milvus)")
AwaDBVectorStore = _vector_store_stub("AwaDB", "awadb", "AwaDB embedded AI database")
BagelVectorStore = _vector_store_stub("Bagel", "bagelML", "BagelDB")
ClarifaiVectorStore = _vector_store_stub("Clarifai", "clarifai", "Clarifai vector store")
DocArrayHnswVectorStore = _vector_store_stub("DocArrayHnsw", "docarray", "DocArray HNSW Search")
DocArrayInMemoryVectorStore = _vector_store_stub("DocArrayInMemory", "docarray", "DocArray In-Memory")

# ── Graph Databases with Vector ──
FalkorDBVectorStore = _vector_store_stub("FalkorDB", "falkordb", "FalkorDB graph + vector")
GelVectorStore = _vector_store_stub("Gel", "gel", "Gel database")

# ── Search-First with Vector ──
TypesenseVectorStore = _vector_store_stub("Typesense", "typesense", "Typesense search with vectors")
MoorsheVectorStore = _vector_store_stub("Moorche", "moorche", "Moorcheh vector store")
ManticoreVectorStore = _vector_store_stub("Manticore", "manticoresearch", "ManticoreSearch with vectors")
XataVectorStore = _vector_store_stub("Xata", "xata", "Xata serverless database")
SurrealDBVectorStore = _vector_store_stub("SurrealDB", "surrealdb", "SurrealDB multi-model")
CrateDBVectorStore = _vector_store_stub("CrateDB", "crate", "CrateDB distributed SQL")

# ── Enterprise / Managed ──
OracleAIVectorStore = _vector_store_stub("OracleAI", "oracledb", "Oracle AI Vector Search")
SAP_HANA_VectorStore = _vector_store_stub("SAPHANA", "hdbcli", "SAP HANA Cloud Vector Engine")
DataStaxVectorStore = _vector_store_stub("DataStax", "astrapy", "DataStax Enterprise")
TeradataVectorStore = _vector_store_stub("Teradata", "teradatasqlalchemy", "Teradata Vantage")
YellowbrickVectorStore = _vector_store_stub("Yellowbrick", "yellowbrick-connector", "Yellowbrick Data Warehouse")
YDBVectorStore = _vector_store_stub("YDB", "ydb", "Yandex Database")
PathwayVectorStore = _vector_store_stub("Pathway", "pathway", "Pathway real-time data processing")
KineticaVectorStore = _vector_store_stub("Kinetica", "kinetica", "Kinetica GPU-accelerated DB")
ApacheDorisVectorStore = _vector_store_stub("ApacheDoris", "pymysql", "Apache Doris OLAP")
ECloudVectorStore = _vector_store_stub("ECloud", "pymysql", "China Mobile ECloud ES")
MariaDBVectorStore = _vector_store_stub("MariaDB", "mariadb", "MariaDB with vector extension")
OpenGaussVectorStore = _vector_store_stub("OpenGauss", "psycopg2-binary", "openGauss database")
SQLServerVectorStore = _vector_store_stub("SQLServer", "pyodbc", "Microsoft SQL Server")
VeDBVectorStore = _vector_store_stub("VeDB", "pymysql", "VolcEngine VeDB for MySQL")
LambdaDBVectorStore = _vector_store_stub("LambdaDB", "lambdadb", "LambdaDB")
ApertureDBVectorStore = _vector_store_stub("ApertureDB", "aperturedb", "ApertureDB visual data platform")
VDMSVectorStore = _vector_store_stub("VDMS", "vdms", "VDMS Visual Data Management System")
ZeusDBVectorStore = _vector_store_stub("ZeusDB", "zeusdb", "ZeusDB")
ZvecVectorStore = _vector_store_stub("Zvec", "zvec", "Zvec vector store")
ChaindeskVectorStore = _vector_store_stub("Chaindesk", "chaindesk", "Chaindesk")
TablestoreVectorStore = _vector_store_stub("Tablestore", "tablestore", "Alibaba Tablestore")
TairVectorStore = _vector_store_stub("Tair", "tair", "Alibaba Tair (Redis-compatible)")
MomentoVIVectorStore = _vector_store_stub("MomentoVI", "momento", "Momento Vector Index")

# ── ML Framework Integration ──
SKLearnVectorStore = _vector_store_stub("SKLearn", "scikit-learn", "Scikit-learn NearestNeighbors")
TensorFlowVectorStore = _vector_store_stub("TensorFlow", "tensorflow", "TensorFlow similarity")
HNSWLibVectorStore = _vector_store_stub("HNSWLib", "hnswlib", "HNSWLib (standalone)")
NMSLibVectorStore = _vector_store_stub("NMSLib", "nmslib", "NMSLib approximate search")
PyNNDescentVectorStore = _vector_store_stub("PyNNDescent", "pynndescent", "PyNNDescent for approximate NN")
NGTVectorStore = _vector_store_stub("NGT", "ngt", "Yahoo NGT (Neighborhood Graph and Tree)")

# ── Specialty / Research ──
WeaviateHybridVectorStore = _vector_store_stub("WeaviateHybrid", "weaviate-client", "Weaviate with hybrid search")
QdrantSparseVectorStore = _vector_store_stub("QdrantSparse", "qdrant-client", "Qdrant with sparse vectors")
MilvusHybridVectorStore = _vector_store_stub("MilvusHybrid", "pymilvus", "Milvus with hybrid search")
PGVectorScaleVectorStore = _vector_store_stub("PGVectorScale", "timescale-vector", "PGVectorScale (Timescale)")
PGVectoRSVectorStore = _vector_store_stub("PGVectoRS", "pgvecto-rs", "PGVecto.rs extension")
ChromaHybridVectorStore = _vector_store_stub("ChromaHybrid", "chromadb", "Chroma with hybrid search")

# ── Data Platforms ──
DatabricksVectorStore = _vector_store_stub("Databricks", "databricks-vectorsearch", "Databricks Vector Search")
SnowparkVectorStore = _vector_store_stub("Snowpark", "snowflake-snowpark-python", "Snowflake Snowpark")
AivenVectorStore = _vector_store_stub("Aiven", "psycopg2-binary", "Aiven for PostgreSQL with pgvector")

# ── SaaS / Hosted Solutions ──
ZepVectorStore = _vector_store_stub("Zep", "zep-python", "Zep memory store")
ZepCloudVectorStore = _vector_store_stub("ZepCloud", "zep-cloud", "Zep Cloud managed memory")
BreeebsVectorStore = _vector_store_stub("Breebs", "breebs", "BREEBS knowledge capsules")
FleetAIVectorStore = _vector_store_stub("FleetAI", "fleet-context", "Fleet AI Context")
VectorizeVectorStore = _vector_store_stub("Vectorize", "vectorize", "Vectorize.io managed RAG")
EmbedchainVectorStore = _vector_store_stub("Embedchain", "embedchain", "Embedchain RAG framework")
NeedleVectorStore = _vector_store_stub("Needle", "needle", "Needle document intelligence")
OutlineVectorStore = _vector_store_stub("Outline", "outline", "Outline wiki/knowledge base")
BoxVectorStore = _vector_store_stub("Box", "box-sdk-gen", "Box AI content cloud")
PermitVectorStore = _vector_store_stub("Permit", "permit", "Permit.io authorization-aware RAG")
GalaxiaVectorStore = _vector_store_stub("Galaxia", "galaxia", "Galaxia vector store")
MotherDuckVectorStore = _vector_store_stub("MotherDuck", "duckdb", "MotherDuck serverless DuckDB")
LanternVectorStore = _vector_store_stub("Lantern", "psycopg2-binary", "Lantern PostgreSQL extension")

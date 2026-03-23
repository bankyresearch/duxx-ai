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

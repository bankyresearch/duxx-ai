"""Embedding providers — convert text to vectors for semantic search."""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Override for batch optimization."""
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 0


class OpenAIEmbedder(Embedder):
    """OpenAI text-embedding-3-small embeddings."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        self.model = model
        self._api_key = api_key
        self._dim = 1536

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key
        import os
        return os.environ.get("OPENAI_API_KEY", "")

    def embed(self, text: str) -> list[float]:
        try:
            import httpx
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._get_key()}", "Content-Type": "application/json"},
                json={"input": text, "model": self.model},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data["data"][0]["embedding"]
            self._dim = len(embedding)
            return embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        try:
            import httpx
            resp = httpx.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._get_key()}", "Content-Type": "application/json"},
                json={"input": texts, "model": self.model},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            return [self.embed(t) for t in texts]


class LocalEmbedder(Embedder):
    """Simple hash-based embedder for testing (no external dependencies).

    Produces deterministic pseudo-embeddings using character trigram hashing.
    NOT suitable for production — use OpenAIEmbedder or sentence-transformers instead.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        # Create a deterministic pseudo-embedding from text hash
        text = text.lower().strip()
        vector = [0.0] * self._dim

        # Use character trigrams hashed into buckets
        for i in range(len(text) - 2):
            trigram = text[i:i + 3]
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
            idx = h % self._dim
            vector[idx] += 1.0

        # Normalize to unit vector
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]

        return vector


class HuggingFaceEmbedder(Embedder):
    """HuggingFace sentence-transformers embedder (local, free).

    Requires: pip install sentence-transformers

    Usage:
        embedder = HuggingFaceEmbedder("all-MiniLM-L6-v2")
        vector = embedder.embed("Hello world")
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        return vecs.tolist()


class CohereEmbedder(Embedder):
    """Cohere embeddings (cloud API).

    Requires: pip install cohere

    Usage:
        embedder = CohereEmbedder(api_key="...")
    """

    def __init__(self, model: str = "embed-english-v3.0", api_key: str = "", input_type: str = "search_document") -> None:
        self.model = model
        self._api_key = api_key
        self._input_type = input_type
        self._dim = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key
        import os
        return os.environ.get("COHERE_API_KEY", "")

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        try:
            import httpx
            resp = httpx.post(
                "https://api.cohere.ai/v1/embed",
                headers={"Authorization": f"Bearer {self._get_key()}", "Content-Type": "application/json"},
                json={"texts": texts, "model": self.model, "input_type": self._input_type},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data["embeddings"]
            if embeddings:
                self._dim = len(embeddings[0])
            return embeddings
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            raise


class VoyageEmbedder(Embedder):
    """Voyage AI embeddings.

    Requires: VOYAGE_API_KEY environment variable

    Usage:
        embedder = VoyageEmbedder(api_key="...")
    """

    def __init__(self, model: str = "voyage-3", api_key: str = "") -> None:
        self.model = model
        self._api_key = api_key
        self._dim = 1024

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key
        import os
        return os.environ.get("VOYAGE_API_KEY", "")

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._get_key()}", "Content-Type": "application/json"},
            json={"input": texts, "model": self.model},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        if embeddings:
            self._dim = len(embeddings[0])
        return embeddings


class NVIDIAEmbedder(Embedder):
    """NVIDIA NIM embeddings. Requires: NVIDIA_API_KEY env var."""
    def __init__(self, model: str = "nvidia/nv-embedqa-e5-v5", api_key: str = "") -> None:
        self.model = model; self._api_key = api_key; self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import os, httpx
        key = self._api_key or os.environ.get("NVIDIA_API_KEY", "")
        resp = httpx.post("https://integrate.api.nvidia.com/v1/embeddings", headers={"Authorization": f"Bearer {key}"}, json={"input": texts, "model": self.model}, timeout=60)
        resp.raise_for_status(); data = resp.json(); embs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        if embs: self._dim = len(embs[0])
        return embs


class JinaEmbedder(Embedder):
    """Jina AI embeddings. Requires: JINA_API_KEY env var."""
    def __init__(self, model: str = "jina-embeddings-v3", api_key: str = "") -> None:
        self.model = model; self._api_key = api_key; self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import os, httpx
        key = self._api_key or os.environ.get("JINA_API_KEY", "")
        resp = httpx.post("https://api.jina.ai/v1/embeddings", headers={"Authorization": f"Bearer {key}"}, json={"input": texts, "model": self.model}, timeout=60)
        resp.raise_for_status(); data = resp.json()
        embs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        if embs: self._dim = len(embs[0])
        return embs


class NomicEmbedder(Embedder):
    """Nomic AI embeddings. Requires: NOMIC_API_KEY env var."""
    def __init__(self, model: str = "nomic-embed-text-v1.5", api_key: str = "") -> None:
        self.model = model; self._api_key = api_key; self._dim = 768
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import os, httpx
        key = self._api_key or os.environ.get("NOMIC_API_KEY", "")
        resp = httpx.post("https://api-atlas.nomic.ai/v1/embedding/text", headers={"Authorization": f"Bearer {key}"}, json={"texts": texts, "model": self.model}, timeout=60)
        resp.raise_for_status(); return resp.json().get("embeddings", [])


class FastEmbedEmbedder(Embedder):
    """FastEmbed (Qdrant) — fast local embeddings. Requires: pip install fastembed"""
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try: from fastembed import TextEmbedding
        except ImportError: raise ImportError("fastembed required: pip install fastembed")
        self._model = TextEmbedding(model); self._dim = 384
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return list(self._model.embed([text]))[0].tolist()
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        results = list(self._model.embed(texts)); return [r.tolist() for r in results]


class BedrockEmbedder(Embedder):
    """AWS Bedrock embeddings (Titan, Cohere). Requires: pip install boto3"""
    def __init__(self, model: str = "amazon.titan-embed-text-v2:0", region: str = "us-east-1") -> None:
        self.model = model; self.region = region; self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]:
        import boto3, json
        client = boto3.client("bedrock-runtime", region_name=self.region)
        resp = client.invoke_model(modelId=self.model, body=json.dumps({"inputText": text}))
        data = json.loads(resp["body"].read()); vec = data.get("embedding", [])
        self._dim = len(vec); return vec
    def embed_many(self, texts: list[str]) -> list[list[float]]: return [self.embed(t) for t in texts]


class AzureEmbedder(Embedder):
    """Azure OpenAI embeddings. Requires: AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT."""
    def __init__(self, deployment: str = "text-embedding-ada-002", api_version: str = "2024-02-01") -> None:
        import os
        self.deployment = deployment; self.api_version = api_version
        self._endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", ""); self._key = os.environ.get("AZURE_OPENAI_API_KEY", ""); self._dim = 1536
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        url = f"{self._endpoint}/openai/deployments/{self.deployment}/embeddings?api-version={self.api_version}"
        resp = httpx.post(url, headers={"api-key": self._key}, json={"input": texts}, timeout=60)
        resp.raise_for_status(); return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda x: x["index"])]


class GoogleEmbedder(Embedder):
    """Google Gemini / Vertex AI embeddings.

    Requires: pip install google-generativeai

    Usage:
        embedder = GoogleEmbedder(api_key="...")
    """

    def __init__(self, model: str = "models/text-embedding-004", api_key: str = "") -> None:
        self.model = model
        self._api_key = api_key
        self._dim = 768

    @property
    def dimension(self) -> int:
        return self._dim

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key
        import os
        return os.environ.get("GOOGLE_API_KEY", "")

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        key = self._get_key()
        results = []
        for text in texts:
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={key}",
                json={"model": self.model, "content": {"parts": [{"text": text}]}},
                timeout=30,
            )
            resp.raise_for_status()
            vec = resp.json()["embedding"]["values"]
            self._dim = len(vec)
            results.append(vec)
        return results

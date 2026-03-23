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

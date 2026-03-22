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

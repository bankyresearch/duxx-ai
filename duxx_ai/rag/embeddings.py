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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OpenAI-Compatible Embedder Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _openai_compatible_embedder_factory(name: str, base_url: str, env_key: str, default_model: str, default_dim: int = 1536) -> type[Embedder]:
    """Factory for creating OpenAI-compatible embedding provider classes."""
    class _Embedder(Embedder):
        __doc__ = f"{name} embeddings (OpenAI-compatible). Set {env_key} env var or pass api_key."
        def __init__(self, model: str = default_model, api_key: str = "") -> None:
            self.model = model; self._api_key = api_key; self._dim = default_dim; self._base_url = base_url
        @property
        def dimension(self) -> int: return self._dim
        def _get_key(self) -> str:
            import os
            return self._api_key or os.environ.get(env_key, "")
        def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
        def embed_many(self, texts: list[str]) -> list[list[float]]:
            import httpx
            resp = httpx.post(f"{self._base_url}/embeddings", headers={"Authorization": f"Bearer {self._get_key()}", "Content-Type": "application/json"}, json={"input": texts, "model": self.model}, timeout=60)
            resp.raise_for_status(); data = resp.json()
            embs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
            if embs: self._dim = len(embs[0])
            return embs
    _Embedder.__name__ = f"{name}Embedder"
    _Embedder.__qualname__ = f"{name}Embedder"
    return _Embedder


# ── Tier 2: OpenAI-compatible API embedders ──
MistralEmbedder = _openai_compatible_embedder_factory("Mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "mistral-embed", 1024)
TogetherEmbedder = _openai_compatible_embedder_factory("Together", "https://api.together.xyz/v1", "TOGETHER_API_KEY", "togethercomputer/m2-bert-80M-8k-retrieval", 768)
FireworksEmbedder = _openai_compatible_embedder_factory("Fireworks", "https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY", "nomic-ai/nomic-embed-text-v1.5", 768)
DeepInfraEmbedder = _openai_compatible_embedder_factory("DeepInfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", "BAAI/bge-large-en-v1.5", 1024)
GroqEmbedder = _openai_compatible_embedder_factory("Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.2-embedding", 768)
OpenRouterEmbedder = _openai_compatible_embedder_factory("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/text-embedding-3-small", 1536)
AnyscaleEmbedder = _openai_compatible_embedder_factory("Anyscale", "https://api.endpoints.anyscale.com/v1", "ANYSCALE_API_KEY", "thenlper/gte-large", 1024)
PerplexityEmbedder = _openai_compatible_embedder_factory("Perplexity", "https://api.perplexity.ai", "PERPLEXITY_API_KEY", "pplx-embed", 1024)
LeptonEmbedder = _openai_compatible_embedder_factory("Lepton", "https://llm.lepton.run/api/v1", "LEPTON_API_KEY", "embedding", 768)
NebiusEmbedder = _openai_compatible_embedder_factory("Nebius", "https://api.studio.nebius.ai/v1", "NEBIUS_API_KEY", "BAAI/bge-en-icl", 1024)
DatabricksEmbedder = _openai_compatible_embedder_factory("Databricks", "https://YOUR_WORKSPACE.databricks.com/serving-endpoints", "DATABRICKS_TOKEN", "databricks-bge-large-en", 1024)
VLLMEmbedder = _openai_compatible_embedder_factory("vLLM", "http://localhost:8000/v1", "", "BAAI/bge-small-en-v1.5", 384)
OllamaEmbedder = _openai_compatible_embedder_factory("Ollama", "http://localhost:11434/v1", "", "nomic-embed-text", 768)
LMStudioEmbedder = _openai_compatible_embedder_factory("LMStudio", "http://localhost:1234/v1", "", "nomic-embed-text-v1.5", 768)
CloudflareEmbedder = _openai_compatible_embedder_factory("Cloudflare", "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "CLOUDFLARE_API_TOKEN", "@cf/baai/bge-small-en-v1.5", 384)
XAIEmbedder = _openai_compatible_embedder_factory("xAI", "https://api.x.ai/v1", "XAI_API_KEY", "v1", 1024)


# ── Tier 3: Non-OpenAI API embedders ──

class AlephAlphaEmbedder(Embedder):
    """Aleph Alpha embeddings (Luminous models). Requires: ALEPH_ALPHA_API_KEY."""
    def __init__(self, model: str = "luminous-base", api_key: str = "") -> None:
        self.model = model; self._api_key = api_key; self._dim = 5120
    @property
    def dimension(self) -> int: return self._dim
    def _get_key(self) -> str:
        import os; return self._api_key or os.environ.get("ALEPH_ALPHA_API_KEY", "")
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx; results = []
        for text in texts:
            resp = httpx.post("https://api.aleph-alpha.com/semantic_embed", headers={"Authorization": f"Bearer {self._get_key()}"}, json={"model": self.model, "prompt": {"type": "text", "data": text}, "representation": "document"}, timeout=30)
            resp.raise_for_status(); results.append(resp.json()["embedding"])
        return results


class UpstageEmbedder(Embedder):
    """Upstage Solar embeddings. Requires: UPSTAGE_API_KEY."""
    def __init__(self, model: str = "solar-embedding-1-large-query", api_key: str = "") -> None:
        self.model = model; self._api_key = api_key; self._dim = 4096
    @property
    def dimension(self) -> int: return self._dim
    def _get_key(self) -> str:
        import os; return self._api_key or os.environ.get("UPSTAGE_API_KEY", "")
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post("https://api.upstage.ai/v1/solar/embeddings", headers={"Authorization": f"Bearer {self._get_key()}"}, json={"input": texts, "model": self.model}, timeout=60)
        resp.raise_for_status()
        return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda x: x["index"])]


class WatsonxEmbedder(Embedder):
    """IBM Watsonx.ai embeddings. Requires: WATSONX_API_KEY + WATSONX_PROJECT_ID."""
    def __init__(self, model: str = "ibm/slate-125m-english-rtrvr", api_key: str = "", project_id: str = "") -> None:
        import os; self.model = model; self._api_key = api_key or os.environ.get("WATSONX_API_KEY", "")
        self._project_id = project_id or os.environ.get("WATSONX_PROJECT_ID", ""); self._dim = 768
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post("https://us-south.ml.cloud.ibm.com/ml/v1/text/embeddings?version=2024-01-01",
            headers={"Authorization": f"Bearer {self._api_key}"}, json={"input": texts, "model_id": self.model, "project_id": self._project_id}, timeout=60)
        resp.raise_for_status(); results = resp.json().get("results", [])
        return [r["embedding"] for r in results]


class OCIEmbedder(Embedder):
    """Oracle Cloud Infrastructure GenAI embeddings. Requires: pip install oci."""
    def __init__(self, model: str = "cohere.embed-english-v3.0", compartment_id: str = "") -> None:
        self.model = model; self._compartment_id = compartment_id; self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        try: import oci
        except ImportError: raise ImportError("oci required: pip install oci")
        config = oci.config.from_file()
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(config)
        resp = client.embed_text(oci.generative_ai_inference.models.EmbedTextDetails(
            inputs=texts, serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.model),
            compartment_id=self._compartment_id, input_type="SEARCH_DOCUMENT",
        ))
        return resp.data.embeddings


class SpacyEmbedder(Embedder):
    """SpaCy word embeddings (local). Requires: pip install spacy + model download."""
    def __init__(self, model: str = "en_core_web_md") -> None:
        try: import spacy
        except ImportError: raise ImportError("spacy required: pip install spacy && python -m spacy download en_core_web_md")
        self._nlp = spacy.load(model); self._dim = self._nlp.vocab.vectors.shape[1] if self._nlp.vocab.vectors.shape else 300
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self._nlp(text).vector.tolist()
    def embed_many(self, texts: list[str]) -> list[list[float]]: return [self.embed(t) for t in texts]


class InfinityEmbedder(Embedder):
    """Infinity (self-hosted embedding server). Requires: running infinity server."""
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", url: str = "http://localhost:7997") -> None:
        self.model = model; self._url = url; self._dim = 384
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(f"{self._url}/embeddings", json={"input": texts, "model": self.model}, timeout=60)
        resp.raise_for_status(); data = resp.json()
        embs = [d["embedding"] for d in sorted(data["data"], key=lambda x: x["index"])]
        if embs: self._dim = len(embs[0])
        return embs


class TEIEmbedder(Embedder):
    """HuggingFace Text Embeddings Inference (TEI) server. Self-hosted."""
    def __init__(self, url: str = "http://localhost:8080") -> None:
        self._url = url; self._dim = 384
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(f"{self._url}/embed", json={"inputs": texts}, timeout=60)
        resp.raise_for_status(); embs = resp.json()
        if embs and isinstance(embs[0], list): self._dim = len(embs[0])
        return embs


class SageMakerEmbedder(Embedder):
    """AWS SageMaker endpoint embeddings. Requires: pip install boto3."""
    def __init__(self, endpoint_name: str, region: str = "us-east-1") -> None:
        self._endpoint = endpoint_name; self._region = region; self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import boto3, json
        client = boto3.client("sagemaker-runtime", region_name=self._region)
        resp = client.invoke_endpoint(EndpointName=self._endpoint, ContentType="application/json", Body=json.dumps({"inputs": texts}))
        data = json.loads(resp["Body"].read()); return data if isinstance(data, list) else data.get("embeddings", [])


class GradientEmbedder(Embedder):
    """Gradient AI embeddings. Requires: GRADIENT_ACCESS_TOKEN."""
    def __init__(self, model: str = "bge-large", api_key: str = "", workspace_id: str = "") -> None:
        import os; self.model = model; self._key = api_key or os.environ.get("GRADIENT_ACCESS_TOKEN", "")
        self._workspace = workspace_id or os.environ.get("GRADIENT_WORKSPACE_ID", ""); self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(f"https://api.gradient.ai/api/embeddings/{self.model}", headers={"Authorization": f"Bearer {self._key}", "x-gradient-workspace-id": self._workspace}, json={"inputs": [{"input": t} for t in texts]}, timeout=60)
        resp.raise_for_status(); return [e["embedding"] for e in resp.json()["embeddings"]]


class MinimaxEmbedder(Embedder):
    """MiniMax embeddings. Requires: MINIMAX_API_KEY."""
    def __init__(self, model: str = "embo-01", api_key: str = "", group_id: str = "") -> None:
        import os; self.model = model; self._key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self._group = group_id or os.environ.get("MINIMAX_GROUP_ID", ""); self._dim = 1536
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(f"https://api.minimax.chat/v1/embeddings?GroupId={self._group}", headers={"Authorization": f"Bearer {self._key}"}, json={"texts": texts, "model": self.model, "type": "db"}, timeout=60)
        resp.raise_for_status(); return resp.json()["vectors"]


class BaichuanEmbedder(Embedder):
    """Baichuan embeddings. Requires: BAICHUAN_API_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("BAICHUAN_API_KEY", ""); self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post("https://api.baichuan-ai.com/v1/embeddings", headers={"Authorization": f"Bearer {self._key}"}, json={"input": texts, "model": "Baichuan-Text-Embedding"}, timeout=60)
        resp.raise_for_status(); return [d["embedding"] for d in resp.json()["data"]]


class QianfanEmbedder(Embedder):
    """Baidu Qianfan embeddings. Requires: QIANFAN_AK + QIANFAN_SK."""
    def __init__(self, model: str = "embedding-v1", ak: str = "", sk: str = "") -> None:
        import os; self.model = model; self._ak = ak or os.environ.get("QIANFAN_AK", "")
        self._sk = sk or os.environ.get("QIANFAN_SK", ""); self._dim = 1024; self._token = ""
    @property
    def dimension(self) -> int: return self._dim
    def _get_token(self) -> str:
        if self._token: return self._token
        import httpx
        resp = httpx.post(f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self._ak}&client_secret={self._sk}")
        self._token = resp.json()["access_token"]; return self._token
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx; token = self._get_token()
        resp = httpx.post(f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{self.model}?access_token={token}", json={"input": texts}, timeout=60)
        resp.raise_for_status(); return [d["embedding"] for d in resp.json()["data"]]


class DashScopeEmbedder(Embedder):
    """Alibaba DashScope (Qwen) embeddings. Requires: DASHSCOPE_API_KEY."""
    def __init__(self, model: str = "text-embedding-v3", api_key: str = "") -> None:
        import os; self.model = model; self._key = api_key or os.environ.get("DASHSCOPE_API_KEY", ""); self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post("https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding", headers={"Authorization": f"Bearer {self._key}"}, json={"model": self.model, "input": {"texts": texts}, "parameters": {"text_type": "document"}}, timeout=60)
        resp.raise_for_status(); return [o["embedding"] for o in resp.json()["output"]["embeddings"]]


class VolcEngineEmbedder(Embedder):
    """Volcengine (ByteDance) embeddings. Requires: VOLC_API_KEY."""
    def __init__(self, model: str = "doubao-embedding", api_key: str = "") -> None:
        import os; self.model = model; self._key = api_key or os.environ.get("VOLC_API_KEY", ""); self._dim = 1024
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post("https://ark.cn-beijing.volces.com/api/v3/embeddings", headers={"Authorization": f"Bearer {self._key}"}, json={"input": texts, "model": self.model}, timeout=60)
        resp.raise_for_status(); return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda x: x["index"])]


class SparkEmbedder(Embedder):
    """iFlyTek Spark embeddings. Requires: SPARK_API_KEY."""
    def __init__(self, api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("SPARK_API_KEY", ""); self._dim = 2560
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx; results = []
        for t in texts:
            resp = httpx.post("https://knowledge-retrieval.cn-huabei-1.volces.com/api/v1/embedding", headers={"Authorization": f"Bearer {self._key}"}, json={"text": t}, timeout=30)
            resp.raise_for_status(); results.append(resp.json()["embedding"])
        return results


class ZhipuEmbedder(Embedder):
    """Zhipu AI (GLM) embeddings. Requires: ZHIPU_API_KEY."""
    def __init__(self, model: str = "embedding-3", api_key: str = "") -> None:
        import os; self.model = model; self._key = api_key or os.environ.get("ZHIPU_API_KEY", ""); self._dim = 2048
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post("https://open.bigmodel.cn/api/paas/v4/embeddings", headers={"Authorization": f"Bearer {self._key}"}, json={"input": texts, "model": self.model}, timeout=60)
        resp.raise_for_status(); return [d["embedding"] for d in sorted(resp.json()["data"], key=lambda x: x["index"])]


class GPT4AllEmbedder(Embedder):
    """GPT4All local embeddings. Requires: pip install gpt4all."""
    def __init__(self, model: str = "all-MiniLM-L6-v2.gguf2.f16.gguf") -> None:
        try: from gpt4all import Embed4All
        except ImportError: raise ImportError("gpt4all required: pip install gpt4all")
        self._model = Embed4All(model); self._dim = 384
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self._model.embed(text)
    def embed_many(self, texts: list[str]) -> list[list[float]]: return [self.embed(t) for t in texts]


class LlamaCppEmbedder(Embedder):
    """Llama.cpp server embeddings (local). Requires: running llama.cpp server."""
    def __init__(self, url: str = "http://localhost:8080") -> None:
        self._url = url; self._dim = 4096
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(f"{self._url}/embedding", json={"content": texts}, timeout=60)
        resp.raise_for_status(); data = resp.json()
        if isinstance(data, list): embs = [d.get("embedding", d) for d in data]
        else: embs = data.get("embedding", [[]])
        if embs and isinstance(embs[0], list): self._dim = len(embs[0])
        return embs if isinstance(embs[0], list) else [embs]


class ClarifaiEmbedder(Embedder):
    """Clarifai embeddings. Requires: CLARIFAI_PAT."""
    def __init__(self, model_id: str = "BAAI-bge-base-en-v15", user_id: str = "clarifai", app_id: str = "main", api_key: str = "") -> None:
        import os; self._key = api_key or os.environ.get("CLARIFAI_PAT", "")
        self._user = user_id; self._app = app_id; self._model = model_id; self._dim = 768
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx; results = []
        for t in texts:
            resp = httpx.post(f"https://api.clarifai.com/v2/users/{self._user}/apps/{self._app}/models/{self._model}/outputs",
                headers={"Authorization": f"Key {self._key}"}, json={"inputs": [{"data": {"text": {"raw": t}}}]}, timeout=30)
            resp.raise_for_status()
            emb = resp.json()["outputs"][0]["data"]["embeddings"][0]["vector"]
            results.append(emb)
        return results


class EdenAIEmbedder(Embedder):
    """Eden AI embeddings (multi-provider). Requires: EDENAI_API_KEY."""
    def __init__(self, provider: str = "openai", api_key: str = "") -> None:
        import os; self._provider = provider; self._key = api_key or os.environ.get("EDENAI_API_KEY", ""); self._dim = 1536
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx; results = []
        for t in texts:
            resp = httpx.post("https://api.edenai.run/v2/text/embeddings", headers={"Authorization": f"Bearer {self._key}"}, json={"providers": self._provider, "texts": [t]}, timeout=30)
            resp.raise_for_status(); results.append(resp.json()[self._provider]["items"][0]["embedding"])
        return results


class PredictionGuardEmbedder(Embedder):
    """Prediction Guard embeddings. Requires: PREDICTIONGUARD_API_KEY."""
    def __init__(self, model: str = "bridgetower-large-itm-mlm-itc", api_key: str = "") -> None:
        import os; self.model = model; self._key = api_key or os.environ.get("PREDICTIONGUARD_API_KEY", ""); self._dim = 512
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self.embed_many([text])[0]
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import httpx; results = []
        for t in texts:
            resp = httpx.post("https://api.predictionguard.com/embeddings", headers={"Authorization": f"Bearer {self._key}"}, json={"model": self.model, "input": [{"text": t}]}, timeout=30)
            resp.raise_for_status(); results.append(resp.json()["data"][0]["embedding"])
        return results


class Model2VecEmbedder(Embedder):
    """Model2Vec static embeddings (ultra-fast, local). Requires: pip install model2vec."""
    def __init__(self, model: str = "minishlab/potion-base-8M") -> None:
        try: from model2vec import StaticModel
        except ImportError: raise ImportError("model2vec required: pip install model2vec")
        self._model = StaticModel.from_pretrained(model); self._dim = self._model.dim
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]: return self._model.encode(text).tolist()
    def embed_many(self, texts: list[str]) -> list[list[float]]: return self._model.encode(texts).tolist()


class OpenVINOEmbedder(Embedder):
    """OpenVINO optimized embeddings (Intel). Requires: pip install optimum[openvino]."""
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from optimum.intel import OVModelForFeatureExtraction
            from transformers import AutoTokenizer
        except ImportError: raise ImportError("optimum[openvino] required: pip install optimum[openvino]")
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = OVModelForFeatureExtraction.from_pretrained(model, export=True); self._dim = 384
    @property
    def dimension(self) -> int: return self._dim
    def embed(self, text: str) -> list[float]:
        import torch
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self._model(**inputs)
        vec = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy().tolist()
        self._dim = len(vec); return vec
    def embed_many(self, texts: list[str]) -> list[list[float]]: return [self.embed(t) for t in texts]

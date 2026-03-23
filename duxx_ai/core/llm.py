"""LLM provider abstraction — supports OpenAI, Anthropic, local models, and custom endpoints."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

import httpx
from pydantic import BaseModel, Field

from duxx_ai.core.message import Conversation, Message, Role, ToolCall
from duxx_ai.core.tool import Tool


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    extra: dict[str, Any] = Field(default_factory=dict)

    def get_api_key(self) -> str:
        """Return the API key, falling back to environment variables."""
        import os
        if self.api_key:
            return self.api_key
        env_map = {
            "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
            "local": "", "google": "GOOGLE_API_KEY", "gemini": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY", "mistral": "MISTRAL_API_KEY",
            "bedrock": "", "deepseek": "DEEPSEEK_API_KEY",
            "together": "TOGETHER_API_KEY", "fireworks": "FIREWORKS_API_KEY",
            "cohere": "COHERE_API_KEY", "perplexity": "PERPLEXITY_API_KEY",
            "xai": "XAI_API_KEY", "cerebras": "CEREBRAS_API_KEY",
            "sambanova": "SAMBANOVA_API_KEY", "ai21": "AI21_API_KEY",
            "nvidia": "NVIDIA_API_KEY", "anyscale": "ANYSCALE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY", "lepton": "LEPTON_API_KEY",
            "replicate": "REPLICATE_API_TOKEN", "ollama": "", "lmstudio": "", "vllm": "",
            "huggingface": "HF_TOKEN", "cloudflare": "CLOUDFLARE_API_TOKEN",
            "moonshot": "MOONSHOT_API_KEY", "zhipu": "ZHIPU_API_KEY",
            "qwen": "DASHSCOPE_API_KEY", "yi": "YI_API_KEY", "nebius": "NEBIUS_API_KEY",
        }
        env_var = env_map.get(self.provider, "")
        return os.environ.get(env_var, "") if env_var else ""


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class LLMProvider(ABC):
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    async def complete(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]: ...


class OpenAIProvider(LLMProvider):
    async def complete(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation.to_dicts(provider="openai"))

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            body["tools"] = [t.to_schema() for t in tools]

        api_key = self.config.get_api_key()
        base = self.config.base_url or "https://api.openai.com/v1"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg = choice["message"]

        tool_calls = []
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                )

        return LLMResponse(
            content=msg.get("content", "") or "",
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            model=data.get("model", self.config.model),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def stream(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(conversation.to_dicts(provider="openai"))

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        api_key = self.config.get_api_key()
        base = self.config.base_url or "https://api.openai.com/v1"
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        if delta.get("content"):
                            yield delta["content"]


class AnthropicProvider(LLMProvider):
    async def complete(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        messages = conversation.to_dicts(provider="anthropic")

        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if system_prompt:
            body["system"] = system_prompt
        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.to_schema()["function"]["parameters"],
                }
                for t in tools
            ]

        api_key = self.config.get_api_key()
        base = self.config.base_url or "https://api.anthropic.com/v1"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{base}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block["type"] == "text":
                content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(id=block["id"], name=block["name"], arguments=block["input"])
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=data.get("usage", {}),
            model=data.get("model", self.config.model),
            finish_reason=data.get("stop_reason", ""),
        )

    async def stream(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        messages = conversation.to_dicts(provider="anthropic")
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if system_prompt:
            body["system"] = system_prompt

        api_key = self.config.get_api_key()
        base = self.config.base_url or "https://api.anthropic.com/v1"
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{base}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    yield delta.get("text", "")
                        except json.JSONDecodeError:
                            continue


class LocalProvider(LLMProvider):
    """Provider for local models served via OpenAI-compatible API (vLLM, llama.cpp, Ollama)."""

    async def complete(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        # Reuse OpenAI provider with custom base_url
        openai = OpenAIProvider(self.config)
        return await openai.complete(conversation, tools, system_prompt)

    async def stream(
        self,
        conversation: Conversation,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        openai = OpenAIProvider(self.config)
        async for chunk in openai.stream(conversation, tools, system_prompt):
            yield chunk


class GoogleProvider(LLMProvider):
    """Google Gemini API provider.

    Usage:
        config = LLMConfig(provider="google", model="gemini-2.0-flash", api_key="...")
        provider = create_provider(config)
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)

    def _get_key(self) -> str:
        import os
        return self.config.api_key or os.environ.get("GOOGLE_API_KEY", "")

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        base = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.config.model}:generateContent?key={self._get_key()}"

        # Convert messages to Gemini format
        contents = []
        for msg in conversation.messages:
            if msg.role == Role.SYSTEM:
                continue  # System prompt handled separately
            role = "user" if msg.role == Role.USER else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        body: dict[str, Any] = {"contents": contents}
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        body["generationConfig"] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_tokens,
        }

        # Add tools if present
        if tools:
            func_declarations = []
            for t in tools:
                schema = t.to_schema()
                func_declarations.append({
                    "name": schema["name"],
                    "description": schema.get("description", ""),
                    "parameters": schema.get("parameters", {}),
                })
            body["tools"] = [{"functionDeclarations": func_declarations}]

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])

        content = ""
        tool_calls = []
        for part in parts:
            if "text" in part:
                content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=f"call_{fc['name']}",
                    name=fc["name"],
                    arguments=fc.get("args", {}),
                ))

        usage = data.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"prompt_tokens": usage.get("promptTokenCount", 0), "completion_tokens": usage.get("candidatesTokenCount", 0), "total_tokens": usage.get("totalTokenCount", 0)},
            model=self.config.model,
        )

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        base = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{self.config.model}:streamGenerateContent?alt=sse&key={self._get_key()}"

        contents = []
        for msg in conversation.messages:
            if msg.role == Role.SYSTEM:
                continue
            role = "user" if msg.role == Role.USER else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        body: dict[str, Any] = {"contents": contents}
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", url, json=body) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = json.loads(line[6:])
                        parts = chunk_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        for part in parts:
                            if "text" in part:
                                yield part["text"]


class GroqProvider(LLMProvider):
    """Groq API provider (OpenAI-compatible, ultra-fast inference).

    Usage:
        config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="...")
    """

    def __init__(self, config: LLMConfig) -> None:
        config.base_url = config.base_url or "https://api.groq.com/openai/v1"
        super().__init__(config)
        self._delegate = OpenAIProvider(config)

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        return await self._delegate.complete(conversation, tools, system_prompt)

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        async for chunk in self._delegate.stream(conversation, tools, system_prompt):
            yield chunk


class MistralProvider(LLMProvider):
    """Mistral AI provider (OpenAI-compatible API).

    Usage:
        config = LLMConfig(provider="mistral", model="mistral-large-latest", api_key="...")
    """

    def __init__(self, config: LLMConfig) -> None:
        config.base_url = config.base_url or "https://api.mistral.ai/v1"
        super().__init__(config)
        self._delegate = OpenAIProvider(config)

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        return await self._delegate.complete(conversation, tools, system_prompt)

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        async for chunk in self._delegate.stream(conversation, tools, system_prompt):
            yield chunk


class BedrockProvider(LLMProvider):
    """AWS Bedrock provider (Claude, Llama, Titan models).

    Requires: pip install boto3

    Usage:
        config = LLMConfig(provider="bedrock", model="anthropic.claude-3-sonnet-20240229-v1:0")
    """

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise ImportError("boto3 is required for Bedrock: pip install boto3")

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        import boto3
        region = self.config.extra.get("region", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        messages = []
        for msg in conversation.messages:
            if msg.role == Role.SYSTEM:
                continue
            role = "user" if msg.role == Role.USER else "assistant"
            messages.append({"role": role, "content": [{"text": msg.content}]})

        body: dict[str, Any] = {
            "messages": messages,
            "inferenceConfig": {"maxTokens": self.config.max_tokens, "temperature": self.config.temperature},
        }
        if system_prompt:
            body["system"] = [{"text": system_prompt}]

        import asyncio
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.converse(modelId=self.config.model, **body),
        )

        output = resp.get("output", {})
        content = ""
        tool_calls = []
        for block in output.get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(id=tu["toolUseId"], name=tu["name"], arguments=tu.get("input", {})))

        usage = resp.get("usage", {})
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"prompt_tokens": usage.get("inputTokens", 0), "completion_tokens": usage.get("outputTokens", 0), "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0)},
            model=self.config.model,
        )

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        # Bedrock streaming via converse_stream
        import boto3, asyncio
        region = self.config.extra.get("region", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        messages = []
        for msg in conversation.messages:
            if msg.role == Role.SYSTEM:
                continue
            role = "user" if msg.role == Role.USER else "assistant"
            messages.append({"role": role, "content": [{"text": msg.content}]})

        body: dict[str, Any] = {
            "messages": messages,
            "inferenceConfig": {"maxTokens": self.config.max_tokens, "temperature": self.config.temperature},
        }
        if system_prompt:
            body["system"] = [{"text": system_prompt}]

        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.converse_stream(modelId=self.config.model, **body),
        )

        for event in resp.get("stream", []):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "text" in delta:
                    yield delta["text"]


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider (OpenAI-compatible).

    Usage:
        config = LLMConfig(provider="deepseek", model="deepseek-chat", api_key="...")
    """

    def __init__(self, config: LLMConfig) -> None:
        config.base_url = config.base_url or "https://api.deepseek.com/v1"
        super().__init__(config)
        self._delegate = OpenAIProvider(config)

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        return await self._delegate.complete(conversation, tools, system_prompt)

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        async for chunk in self._delegate.stream(conversation, tools, system_prompt):
            yield chunk


class TogetherProvider(LLMProvider):
    """Together AI provider (OpenAI-compatible).

    Usage:
        config = LLMConfig(provider="together", model="meta-llama/Llama-3-70b-chat-hf", api_key="...")
    """

    def __init__(self, config: LLMConfig) -> None:
        config.base_url = config.base_url or "https://api.together.xyz/v1"
        super().__init__(config)
        self._delegate = OpenAIProvider(config)

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        return await self._delegate.complete(conversation, tools, system_prompt)

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        async for chunk in self._delegate.stream(conversation, tools, system_prompt):
            yield chunk


class FireworksProvider(LLMProvider):
    """Fireworks AI provider (OpenAI-compatible).

    Usage:
        config = LLMConfig(provider="fireworks", model="accounts/fireworks/models/llama-v3p1-70b-instruct", api_key="...")
    """

    def __init__(self, config: LLMConfig) -> None:
        config.base_url = config.base_url or "https://api.fireworks.ai/inference/v1"
        super().__init__(config)
        self._delegate = OpenAIProvider(config)

    async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
        return await self._delegate.complete(conversation, tools, system_prompt)

    async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
        async for chunk in self._delegate.stream(conversation, tools, system_prompt):
            yield chunk


def _openai_compatible_factory(name: str, base_url: str, env_key: str) -> type[LLMProvider]:
    """Factory for creating OpenAI-compatible provider classes."""
    class _Provider(LLMProvider):
        __doc__ = f"{name} provider (OpenAI-compatible). Set {env_key} env var or pass api_key."
        def __init__(self, config: LLMConfig) -> None:
            config.base_url = config.base_url or base_url
            super().__init__(config)
            self._delegate = OpenAIProvider(config)
        async def complete(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> LLMResponse:
            return await self._delegate.complete(conversation, tools, system_prompt)
        async def stream(self, conversation: Conversation, tools: list[Tool] | None = None, system_prompt: str = "") -> AsyncIterator[str]:
            async for chunk in self._delegate.stream(conversation, tools, system_prompt):
                yield chunk
    _Provider.__name__ = f"{name}Provider"
    _Provider.__qualname__ = f"{name}Provider"
    return _Provider


# Generate 20+ OpenAI-compatible providers
CohereProvider = _openai_compatible_factory("Cohere", "https://api.cohere.ai/v2", "COHERE_API_KEY")
PerplexityProvider = _openai_compatible_factory("Perplexity", "https://api.perplexity.ai", "PERPLEXITY_API_KEY")
XAIProvider = _openai_compatible_factory("xAI", "https://api.x.ai/v1", "XAI_API_KEY")
CerebrasProvider = _openai_compatible_factory("Cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY")
SambaNovaProvider = _openai_compatible_factory("SambaNova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY")
AI21Provider = _openai_compatible_factory("AI21", "https://api.ai21.com/studio/v1", "AI21_API_KEY")
NVIDIAProvider = _openai_compatible_factory("NVIDIA", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY")
AnyscaleProvider = _openai_compatible_factory("Anyscale", "https://api.endpoints.anyscale.com/v1", "ANYSCALE_API_KEY")
OpenRouterProvider = _openai_compatible_factory("OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
LeptonProvider = _openai_compatible_factory("Lepton", "https://llm.lepton.run/api/v1", "LEPTON_API_KEY")
ReplicateProvider = _openai_compatible_factory("Replicate", "https://openai-proxy.replicate.com/v1", "REPLICATE_API_TOKEN")
OllamaProvider = _openai_compatible_factory("Ollama", "http://localhost:11434/v1", "")
LMStudioProvider = _openai_compatible_factory("LMStudio", "http://localhost:1234/v1", "")
VLLMProvider = _openai_compatible_factory("vLLM", "http://localhost:8000/v1", "")
HuggingFaceProvider = _openai_compatible_factory("HuggingFace", "https://api-inference.huggingface.co/v1", "HF_TOKEN")
CloudflareProvider = _openai_compatible_factory("Cloudflare", "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "CLOUDFLARE_API_TOKEN")
MoonshotProvider = _openai_compatible_factory("Moonshot", "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY")
ZhipuProvider = _openai_compatible_factory("Zhipu", "https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY")
QwenProvider = _openai_compatible_factory("Qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY")
YiProvider = _openai_compatible_factory("Yi", "https://api.lingyiwanwu.com/v1", "YI_API_KEY")
NebiusProvider = _openai_compatible_factory("Nebius", "https://api.studio.nebius.ai/v1", "NEBIUS_API_KEY")


PROVIDERS: dict[str, type[LLMProvider]] = {
    # Tier 1 — Custom implementations
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "local": LocalProvider,
    "google": GoogleProvider,
    "gemini": GoogleProvider,
    "bedrock": BedrockProvider,
    # Tier 2 — OpenAI-compatible (custom base_url)
    "groq": GroqProvider,
    "mistral": MistralProvider,
    "deepseek": DeepSeekProvider,
    "together": TogetherProvider,
    "fireworks": FireworksProvider,
    "cohere": CohereProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
    "cerebras": CerebrasProvider,
    "sambanova": SambaNovaProvider,
    "ai21": AI21Provider,
    "nvidia": NVIDIAProvider,
    "anyscale": AnyscaleProvider,
    "openrouter": OpenRouterProvider,
    "lepton": LeptonProvider,
    "replicate": ReplicateProvider,
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider,
    "vllm": VLLMProvider,
    "huggingface": HuggingFaceProvider,
    "cloudflare": CloudflareProvider,
    "moonshot": MoonshotProvider,
    "zhipu": ZhipuProvider,
    "qwen": QwenProvider,
    "yi": YiProvider,
    "nebius": NebiusProvider,
}


def create_provider(config: LLMConfig) -> LLMProvider:
    cls = PROVIDERS.get(config.provider)
    if cls is None:
        raise ValueError(f"Unknown provider: {config.provider}. Available: {list(PROVIDERS)}")
    return cls(config)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Response Cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import hashlib
import time


class LLMCache:
    """In-memory LLM response cache with TTL.

    Usage:
        cache = LLMCache(ttl_seconds=300)
        cached = cache.get(conversation, tools, system_prompt)
        if cached:
            return cached
        response = await provider.complete(...)
        cache.set(conversation, tools, system_prompt, response)
    """

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: dict[str, tuple[float, LLMResponse]] = {}
        self.hits = 0
        self.misses = 0

    def _make_key(self, conversation: Conversation, tools: list[Any] | None, system_prompt: str) -> str:
        parts = [system_prompt]
        for msg in conversation.messages[-10:]:  # Last 10 messages for key
            parts.append(f"{msg.role}:{msg.content or ''}")
        if tools:
            parts.append(str(sorted(t.name for t in tools if hasattr(t, "name"))))
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, conversation: Conversation, tools: list[Any] | None = None, system_prompt: str = "") -> LLMResponse | None:
        key = self._make_key(conversation, tools, system_prompt)
        entry = self._store.get(key)
        if entry:
            ts, response = entry
            if time.time() - ts < self.ttl:
                self.hits += 1
                return response
            del self._store[key]
        self.misses += 1
        return None

    def set(self, conversation: Conversation, tools: list[Any] | None, system_prompt: str, response: LLMResponse) -> None:
        key = self._make_key(conversation, tools, system_prompt)
        if len(self._store) >= self.max_entries:
            # Evict oldest
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]
        self._store[key] = (time.time(), response)

    def clear(self) -> None:
        self._store.clear()

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rate Limiter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import asyncio as _asyncio


class RateLimiter:
    """Token-bucket rate limiter for LLM API calls.

    Usage:
        limiter = RateLimiter(max_requests_per_minute=60)
        await limiter.acquire()  # blocks if rate exceeded
        response = await provider.complete(...)
    """

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        self.max_rpm = max_requests_per_minute
        self._tokens = float(max_requests_per_minute)
        self._max_tokens = float(max_requests_per_minute)
        self._refill_rate = max_requests_per_minute / 60.0  # tokens per second
        self._last_refill = time.time()
        self._lock = _asyncio.Lock()
        self.total_requests = 0
        self.total_waits = 0

    async def acquire(self) -> None:
        """Acquire a rate limit token. Blocks if rate exceeded."""
        async with self._lock:
            self._refill()
            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._refill_rate
                self.total_waits += 1
                await _asyncio.sleep(wait_time)
                self._refill()
            self._tokens -= 1.0
            self.total_requests += 1

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cached + Rate-Limited Provider Wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CachedProvider(LLMProvider):
    """Wraps any LLMProvider with caching and rate limiting.

    Usage:
        base = create_provider(config)
        provider = CachedProvider(base, cache_ttl=300, rate_limit_rpm=60)
        response = await provider.complete(conversation)
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache_ttl: int = 300,
        rate_limit_rpm: int = 0,
    ) -> None:
        self._provider = provider
        self.config = provider.config
        self.cache = LLMCache(ttl_seconds=cache_ttl) if cache_ttl > 0 else None
        self.limiter = RateLimiter(rate_limit_rpm) if rate_limit_rpm > 0 else None

    async def complete(
        self,
        conversation: Conversation,
        tools: list[Any] | None = None,
        system_prompt: str = "",
    ) -> LLMResponse:
        # Check cache first
        if self.cache:
            cached = self.cache.get(conversation, tools, system_prompt)
            if cached:
                return cached

        # Rate limit
        if self.limiter:
            await self.limiter.acquire()

        response = await self._provider.complete(conversation, tools, system_prompt)

        # Cache the response
        if self.cache:
            self.cache.set(conversation, tools, system_prompt, response)

        return response

    async def stream(
        self,
        conversation: Conversation,
        tools: list[Any] | None = None,
        system_prompt: str = "",
    ) -> AsyncIterator[str]:
        if self.limiter:
            await self.limiter.acquire()
        async for token in self._provider.stream(conversation, tools, system_prompt):
            yield token


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Structured Output Helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from pydantic import BaseModel as _PydanticModel


def with_structured_output(
    provider: LLMProvider,
    schema: type[_PydanticModel],
    conversation: Conversation,
    system_prompt: str = "",
    max_retries: int = 2,
) -> _PydanticModel:
    """Call LLM with schema enforcement and parse into a Pydantic model.

    Injects the schema into the prompt and parses the response.
    Retries on parse failure with error feedback.

    Usage:
        class Analysis(BaseModel):
            sentiment: str
            score: float

        result = await with_structured_output(provider, Analysis, conversation)
        print(result.sentiment, result.score)
    """
    # This is a sync helper that returns a coroutine — use with await
    raise NotImplementedError("Use with_structured_output_async instead")


async def with_structured_output_async(
    provider: LLMProvider,
    schema: type[_PydanticModel],
    conversation: Conversation,
    system_prompt: str = "",
    max_retries: int = 2,
) -> _PydanticModel:
    """Async version of with_structured_output."""
    from duxx_ai.core.parsers import PydanticOutputParser

    parser = PydanticOutputParser(schema)
    instructions = parser.get_format_instructions()

    # Inject schema instructions into system prompt
    enhanced_prompt = f"{system_prompt}\n\n{instructions}" if system_prompt else instructions

    last_error = ""
    for attempt in range(max_retries + 1):
        prompt = enhanced_prompt
        if last_error and attempt > 0:
            prompt += f"\n\nPrevious attempt failed: {last_error}. Please fix the JSON format."

        response = await provider.complete(conversation, system_prompt=prompt)

        try:
            return parser.parse(response.content or "")
        except Exception as e:
            last_error = str(e)

    raise ValueError(f"Failed to parse structured output after {max_retries + 1} attempts: {last_error}")

"""Fine-tuning model registry, chat templates, GGUF quantization types, and job manager.

This module provides:
- MODEL_REGISTRY: Supported base models with metadata
- CHAT_TEMPLATES: Formatting templates (Alpaca, ChatML, Llama-3, Phi-3)
- GGUF_QUANT_METHODS: All 13 GGUF quantization types
- FineTuneJob / FineTuneJobManager: Async job tracking for training runs
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Model Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModelInfo(BaseModel):
    """Metadata for a supported base model."""
    id: str
    name: str
    family: str
    params: str
    context_length: int
    recommended_lora_r: int = 16
    recommended_lora_alpha: int = 32
    chat_template: str = "chatml"
    license: str = "apache-2.0"
    provider: str = "unsloth"
    tier: str = "open_source"  # open_source | cloud


MODEL_REGISTRY: dict[str, ModelInfo] = {
    # ── Qwen 2.5 Family ──
    "qwen2.5-7b": ModelInfo(
        id="unsloth/Qwen2.5-7B", name="Qwen 2.5 7B", family="Qwen",
        params="7B", context_length=32768, chat_template="chatml",
        recommended_lora_r=16, recommended_lora_alpha=32,
    ),
    "qwen2.5-14b": ModelInfo(
        id="unsloth/Qwen2.5-14B", name="Qwen 2.5 14B", family="Qwen",
        params="14B", context_length=32768, chat_template="chatml",
        recommended_lora_r=32, recommended_lora_alpha=64,
    ),
    "qwen2.5-72b": ModelInfo(
        id="unsloth/Qwen2.5-72B-bnb-4bit", name="Qwen 2.5 72B", family="Qwen",
        params="72B", context_length=32768, chat_template="chatml",
        recommended_lora_r=64, recommended_lora_alpha=128, tier="cloud",
    ),
    "qwen2.5-coder-7b": ModelInfo(
        id="unsloth/Qwen2.5-Coder-7B", name="Qwen 2.5 Coder 7B", family="Qwen",
        params="7B", context_length=32768, chat_template="chatml",
    ),
    # ── LLaMA 3 Family ──
    "llama3-8b": ModelInfo(
        id="unsloth/Meta-Llama-3.1-8B", name="LLaMA 3.1 8B", family="LLaMA",
        params="8B", context_length=8192, chat_template="llama3",
        license="llama3",
    ),
    "llama3-70b": ModelInfo(
        id="unsloth/Meta-Llama-3.1-70B-bnb-4bit", name="LLaMA 3.1 70B", family="LLaMA",
        params="70B", context_length=8192, chat_template="llama3",
        recommended_lora_r=64, recommended_lora_alpha=128, license="llama3", tier="cloud",
    ),
    # ── Mistral Family ──
    "mistral-7b": ModelInfo(
        id="unsloth/mistral-7b-bnb-4bit", name="Mistral 7B", family="Mistral",
        params="7B", context_length=8192, chat_template="chatml",
    ),
    "mixtral-8x7b": ModelInfo(
        id="unsloth/Mixtral-8x7B-bnb-4bit", name="Mixtral 8x7B", family="Mistral",
        params="46.7B", context_length=32768, chat_template="chatml",
        recommended_lora_r=32, recommended_lora_alpha=64, tier="cloud",
    ),
    # ── Phi Family ──
    "phi3.5-mini": ModelInfo(
        id="unsloth/Phi-3.5-mini-instruct", name="Phi-3.5 Mini", family="Phi",
        params="3.8B", context_length=4096, chat_template="phi3",
        recommended_lora_r=16, recommended_lora_alpha=32, license="mit",
    ),
    "phi4-14b": ModelInfo(
        id="unsloth/phi-4-bnb-4bit", name="Phi-4 14B", family="Phi",
        params="14B", context_length=16384, chat_template="phi3",
        recommended_lora_r=32, recommended_lora_alpha=64, license="mit",
    ),
    # ── Gemma Family ──
    "gemma2-9b": ModelInfo(
        id="unsloth/gemma-2-9b-bnb-4bit", name="Gemma 2 9B", family="Gemma",
        params="9B", context_length=8192, chat_template="chatml",
        license="gemma",
    ),
    # ── DeepSeek Family ──
    "deepseek-r1-8b": ModelInfo(
        id="unsloth/DeepSeek-R1-Distill-Qwen-7B", name="DeepSeek R1 8B", family="DeepSeek",
        params="7B", context_length=32768, chat_template="chatml",
    ),
}


def get_model_info(model_key: str) -> ModelInfo | None:
    """Get model info by registry key."""
    return MODEL_REGISTRY.get(model_key)


def list_models(tier: str | None = None) -> list[dict[str, Any]]:
    """List all registered models, optionally filtered by tier."""
    result = []
    for key, info in MODEL_REGISTRY.items():
        if tier and info.tier != tier:
            continue
        result.append({"key": key, **info.model_dump()})
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Chat Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

CHATML_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

PHI3_TEMPLATE = """<|system|>
{system}<|end|>
<|user|>
{instruction}<|end|>
<|assistant|>
{output}<|end|>"""

CHAT_TEMPLATES: dict[str, str] = {
    "alpaca": ALPACA_TEMPLATE,
    "chatml": CHATML_TEMPLATE,
    "llama3": LLAMA3_TEMPLATE,
    "phi3": PHI3_TEMPLATE,
}


def format_sample(
    template_name: str,
    instruction: str,
    output: str,
    input_text: str = "",
    system: str = "You are a helpful AI assistant.",
) -> str:
    """Format a single training sample using the specified chat template."""
    template = CHAT_TEMPLATES.get(template_name, CHATML_TEMPLATE)
    return template.format(
        instruction=instruction,
        output=output,
        input=input_text,
        system=system,
    )


def format_dataset_for_training(
    samples: list[dict[str, Any]],
    template_name: str = "chatml",
    system_prompt: str = "You are a helpful AI assistant.",
) -> list[dict[str, str]]:
    """Format a list of raw samples into training-ready text using a chat template."""
    formatted = []
    for sample in samples:
        if "messages" in sample:
            # Chat format: extract user/assistant turns
            user_msg = ""
            assistant_msg = ""
            for msg in sample["messages"]:
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    assistant_msg = msg.get("content", "")
            if user_msg and assistant_msg:
                text = format_sample(template_name, user_msg, assistant_msg, system=system_prompt)
                formatted.append({"text": text})
        elif "instruction" in sample:
            text = format_sample(
                template_name,
                sample["instruction"],
                sample.get("output", ""),
                sample.get("input", ""),
                system=system_prompt,
            )
            formatted.append({"text": text})
        elif "text" in sample:
            formatted.append({"text": sample["text"]})
    return formatted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GGUF Quantization Methods
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GGUFQuantMethod(BaseModel):
    """GGUF quantization method info."""
    name: str
    bits: float
    description: str
    size_ratio: float  # Approximate size relative to f16 (1.0 = same as f16)
    quality: str  # low, medium, high, very_high


GGUF_QUANT_METHODS: dict[str, GGUFQuantMethod] = {
    "q2_k": GGUFQuantMethod(
        name="Q2_K", bits=2.5, size_ratio=0.16,
        description="Uses Q4_K for attention.vw and feed_forward.w2, Q2_K for others. Smallest but lowest quality.",
        quality="low",
    ),
    "q3_k_s": GGUFQuantMethod(
        name="Q3_K_S", bits=3.0, size_ratio=0.19,
        description="Uses Q3_K for all tensors. Small size, lower quality.",
        quality="low",
    ),
    "q3_k_m": GGUFQuantMethod(
        name="Q3_K_M", bits=3.5, size_ratio=0.22,
        description="Uses Q4_K for attention.wv, attention.wo, and feed_forward.w2, else Q3_K.",
        quality="medium",
    ),
    "q3_k_l": GGUFQuantMethod(
        name="Q3_K_L", bits=3.5, size_ratio=0.24,
        description="Uses Q5_K for attention.wv, attention.wo, and feed_forward.w2, else Q3_K.",
        quality="medium",
    ),
    "q4_0": GGUFQuantMethod(
        name="Q4_0", bits=4.0, size_ratio=0.25,
        description="Original 4-bit quantization method. Good balance of size and quality.",
        quality="medium",
    ),
    "q4_1": GGUFQuantMethod(
        name="Q4_1", bits=4.5, size_ratio=0.28,
        description="Higher accuracy than Q4_0, quicker inference than Q5 models.",
        quality="medium",
    ),
    "q4_k_s": GGUFQuantMethod(
        name="Q4_K_S", bits=4.5, size_ratio=0.28,
        description="Uses Q4_K for all tensors. Good quality, reasonable size.",
        quality="medium",
    ),
    "q4_k_m": GGUFQuantMethod(
        name="Q4_K_M", bits=4.8, size_ratio=0.30,
        description="Uses Q6_K for half of attention.wv and feed_forward.w2, else Q4_K. Recommended default.",
        quality="high",
    ),
    "q5_0": GGUFQuantMethod(
        name="Q5_0", bits=5.0, size_ratio=0.31,
        description="Higher accuracy, higher resource usage and slower inference.",
        quality="high",
    ),
    "q5_1": GGUFQuantMethod(
        name="Q5_1", bits=5.5, size_ratio=0.34,
        description="Even higher accuracy, resource usage and slower inference.",
        quality="high",
    ),
    "q5_k_s": GGUFQuantMethod(
        name="Q5_K_S", bits=5.5, size_ratio=0.34,
        description="Uses Q5_K for all tensors. High quality.",
        quality="high",
    ),
    "q5_k_m": GGUFQuantMethod(
        name="Q5_K_M", bits=5.7, size_ratio=0.36,
        description="Uses Q6_K for half of attention.wv and feed_forward.w2, else Q5_K.",
        quality="very_high",
    ),
    "q6_k": GGUFQuantMethod(
        name="Q6_K", bits=6.5, size_ratio=0.41,
        description="Uses Q8_K for all tensors. Very high quality, larger size.",
        quality="very_high",
    ),
    "q8_0": GGUFQuantMethod(
        name="Q8_0", bits=8.0, size_ratio=0.50,
        description="Almost indistinguishable from float16. High resource use. Not recommended for most.",
        quality="very_high",
    ),
}


def list_quant_methods() -> list[dict[str, Any]]:
    """List all GGUF quantization methods."""
    return [{"key": k, **v.model_dump()} for k, v in GGUF_QUANT_METHODS.items()]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Job Manager — async training job tracking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JobStatus(str, Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FineTuneJob(BaseModel):
    """Represents a fine-tuning job with status tracking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0  # 0-100
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    best_loss: float = float("inf")
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    error_message: str = ""
    # Config
    base_model: str = ""
    method: str = ""
    dataset_samples: int = 0
    # Timestamps
    created_at: float = Field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    # Results
    output_path: str = ""
    exported_formats: list[str] = Field(default_factory=list)
    eval_scores: dict[str, float] = Field(default_factory=dict)


class FineTuneJobManager:
    """Manages fine-tuning jobs with status tracking (in-memory)."""

    def __init__(self) -> None:
        self.jobs: dict[str, FineTuneJob] = {}

    def create_job(
        self,
        name: str,
        base_model: str,
        method: str = "lora",
        dataset_samples: int = 0,
        total_epochs: int = 3,
    ) -> FineTuneJob:
        """Create a new fine-tuning job."""
        job = FineTuneJob(
            name=name,
            base_model=base_model,
            method=method,
            dataset_samples=dataset_samples,
            total_epochs=total_epochs,
        )
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> FineTuneJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[FineTuneJob]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update_job(self, job_id: str, **kwargs: Any) -> FineTuneJob | None:
        job = self.jobs.get(job_id)
        if job:
            for k, v in kwargs.items():
                if hasattr(job, k):
                    setattr(job, k, v)
        return job

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.PREPARING, JobStatus.TRAINING):
            job.status = JobStatus.CANCELLED
            return True
        return False

    def delete_job(self, job_id: str) -> bool:
        if job_id in self.jobs:
            del self.jobs[job_id]
            return True
        return False


# Global job manager instance
job_manager = FineTuneJobManager()

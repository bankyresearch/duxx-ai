"""Fine-tuning pipeline — train, evaluate, and deploy domain-specific agent models.

Inspired by Unsloth's efficient fine-tuning approach:
- LoRA/QLoRA for parameter-efficient training
- Automatic dataset preparation from agent traces
- Integrated evaluation and deployment
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    source: str = ""  # path to JSONL, or "traces" to use agent traces
    format: str = "chat"  # chat, instruction, completion
    max_samples: int | None = None
    train_split: float = 0.9
    seed: int = 42


class TrainingConfig(BaseModel):
    base_model: str = "unsloth/Qwen2.5-7B"
    output_dir: str = "./duxx_ai-finetuned"
    method: str = "lora"  # lora, qlora, full
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # Training parameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    load_in_4bit: bool = True
    # Hardware
    device: str = "auto"
    num_gpus: int = 1


class TrainingMetrics(BaseModel):
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used_gb: float = 0.0
    elapsed_seconds: float = 0.0


class TrainingResult(BaseModel):
    model_path: str = ""
    metrics_history: list[TrainingMetrics] = Field(default_factory=list)
    final_loss: float = 0.0
    total_steps: int = 0
    total_time_seconds: float = 0.0
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class EvalConfig(BaseModel):
    eval_dataset: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["loss", "accuracy", "f1"])
    num_samples: int = 100


class EvalResult(BaseModel):
    scores: dict[str, float] = Field(default_factory=dict)
    samples_evaluated: int = 0
    model_path: str = ""


class TraceToDataset:
    """Convert agent execution traces into fine-tuning datasets."""

    @staticmethod
    def from_traces(traces_path: str, output_path: str, format: str = "chat") -> int:
        """Convert trace JSONL to training dataset JSONL."""
        count = 0
        traces_file = Path(traces_path)
        if not traces_file.exists():
            logger.warning(f"Traces file not found: {traces_path}")
            return 0

        with open(traces_path) as fin, open(output_path, "w") as fout:
            for line in fin:
                trace = json.loads(line)
                samples = TraceToDataset._trace_to_samples(trace, format)
                for sample in samples:
                    fout.write(json.dumps(sample) + "\n")
                    count += 1

        logger.info(f"Generated {count} training samples from traces")
        return count

    @staticmethod
    def _trace_to_samples(trace: dict[str, Any], format: str) -> list[dict[str, Any]]:
        samples = []
        spans = trace.get("spans", [])

        if format == "chat":
            messages = []
            for span in spans:
                attrs = span.get("attributes", {})
                if "input" in attrs:
                    messages.append({"role": "user", "content": attrs["input"]})
                if "output" in attrs:
                    messages.append({"role": "assistant", "content": attrs["output"]})
                if "tool.name" in attrs:
                    messages.append({
                        "role": "assistant",
                        "content": f"<tool_call>{attrs['tool.name']}({attrs.get('tool.args', '')})</tool_call>",
                    })
            if messages:
                samples.append({"messages": messages})

        elif format == "instruction":
            for span in spans:
                attrs = span.get("attributes", {})
                if "input" in attrs and "output" in attrs:
                    samples.append({
                        "instruction": attrs["input"],
                        "output": attrs["output"],
                    })

        return samples

    @staticmethod
    def from_conversations(conversations: list[dict[str, Any]], output_path: str) -> int:
        """Convert raw conversation dicts to training format."""
        count = 0
        with open(output_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")
                count += 1
        return count


class FineTunePipeline:
    """End-to-end fine-tuning pipeline: data prep -> train -> eval -> export."""

    def __init__(
        self,
        training_config: TrainingConfig | None = None,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        self.training_config = training_config or TrainingConfig()
        self.dataset_config = dataset_config or DatasetConfig()
        self.result: TrainingResult | None = None

    def prepare_dataset(self) -> str:
        """Prepare dataset from configured source. Returns path to processed dataset."""
        if self.dataset_config.source == "traces":
            output = f"{self.training_config.output_dir}/dataset.jsonl"
            Path(self.training_config.output_dir).mkdir(parents=True, exist_ok=True)
            count = TraceToDataset.from_traces("traces.jsonl", output, self.dataset_config.format)
            logger.info(f"Prepared {count} training samples")
            return output
        return self.dataset_config.source

    def train(self, dataset_path: str | None = None) -> TrainingResult:
        """Run fine-tuning. Requires `pip install duxx_ai[finetune]`."""
        try:
            return self._train_with_unsloth(dataset_path)
        except ImportError:
            logger.info("Unsloth not available, falling back to HuggingFace PEFT")
            return self._train_with_peft(dataset_path)

    def _train_with_unsloth(self, dataset_path: str | None) -> TrainingResult:
        """Train using Unsloth for maximum efficiency."""
        from unsloth import FastLanguageModel  # type: ignore

        cfg = self.training_config
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
        )

        from datasets import load_dataset  # type: ignore
        from trl import SFTTrainer  # type: ignore
        from transformers import TrainingArguments  # type: ignore

        ds_path = dataset_path or self.prepare_dataset()
        dataset = load_dataset("json", data_files=ds_path, split="train")

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            fp16=cfg.fp16,
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=cfg.max_seq_length,
        )

        start = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start

        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        self.result = TrainingResult(
            model_path=cfg.output_dir,
            final_loss=train_result.training_loss,
            total_steps=train_result.global_step,
            total_time_seconds=elapsed,
            config=cfg,
        )
        return self.result

    def _train_with_peft(self, dataset_path: str | None) -> TrainingResult:
        """Fallback training using HuggingFace PEFT."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore
        from peft import LoraConfig, get_peft_model  # type: ignore
        from trl import SFTTrainer  # type: ignore
        from datasets import load_dataset  # type: ignore

        cfg = self.training_config

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            device_map=cfg.device,
            load_in_4bit=cfg.load_in_4bit,
        )

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
        )
        model = get_peft_model(model, lora_config)

        ds_path = dataset_path or self.prepare_dataset()
        dataset = load_dataset("json", data_files=ds_path, split="train")

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            fp16=cfg.fp16,
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=cfg.max_seq_length,
        )

        start = time.time()
        train_result = trainer.train()
        elapsed = time.time() - start

        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)

        self.result = TrainingResult(
            model_path=cfg.output_dir,
            final_loss=train_result.training_loss,
            total_steps=train_result.global_step,
            total_time_seconds=elapsed,
            config=cfg,
        )
        return self.result

    def evaluate(self, eval_config: EvalConfig | None = None) -> EvalResult:
        """Evaluate the fine-tuned model on an evaluation dataset.

        Loads the eval dataset, runs the model to compute loss, and returns
        metrics. Falls back gracefully if ML dependencies are missing.
        """
        if self.result is None:
            raise ValueError("No trained model. Run train() first.")

        cfg = eval_config or EvalConfig()

        # If no eval dataset is configured, return training loss as baseline
        if not cfg.eval_dataset:
            logger.warning("No eval_dataset specified; returning training loss only.")
            return EvalResult(
                scores={"loss": self.result.final_loss},
                samples_evaluated=0,
                model_path=self.result.model_path,
            )

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            from datasets import load_dataset  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(self.result.model_path)
            model = AutoModelForCausalLM.from_pretrained(
                self.result.model_path, device_map="auto"
            )
            model.eval()

            dataset = load_dataset("json", data_files=cfg.eval_dataset, split="train")
            if cfg.num_samples and len(dataset) > cfg.num_samples:
                dataset = dataset.select(range(cfg.num_samples))

            total_loss = 0.0
            count = 0

            with torch.no_grad():
                for sample in dataset:
                    # Build input text from chat messages or raw text
                    if "messages" in sample:
                        text = " ".join(m.get("content", "") for m in sample["messages"])
                    elif "text" in sample:
                        text = sample["text"]
                    elif "instruction" in sample and "output" in sample:
                        text = sample["instruction"] + " " + sample["output"]
                    else:
                        continue

                    encodings = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.training_config.max_seq_length,
                    ).to(model.device)

                    outputs = model(**encodings, labels=encodings["input_ids"])
                    total_loss += outputs.loss.item()
                    count += 1

            avg_loss = total_loss / count if count > 0 else 0.0
            scores: dict[str, float] = {"loss": avg_loss}
            logger.info(f"Evaluation complete: avg_loss={avg_loss:.4f} over {count} samples")

            return EvalResult(
                scores=scores,
                samples_evaluated=count,
                model_path=self.result.model_path,
            )

        except ImportError as e:
            logger.warning(
                f"ML dependencies not available for evaluation ({e}). "
                "Install with: pip install duxx_ai[finetune]. "
                "Returning training loss only."
            )
            return EvalResult(
                scores={"loss": self.result.final_loss},
                samples_evaluated=0,
                model_path=self.result.model_path,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvalResult(
                scores={"loss": self.result.final_loss, "eval_error": -1.0},
                samples_evaluated=0,
                model_path=self.result.model_path,
            )

    def export(self, format: str = "safetensors", quantize: str | None = None) -> str:
        """Export the fine-tuned model in the specified format.

        For 'safetensors' format, returns the model directory containing
        the saved model files.

        For 'gguf' format, logs a warning that llama.cpp conversion must
        be done externally (e.g. via convert-hf-to-gguf.py) and returns
        the safetensors model path so downstream tools know where the
        source weights are.
        """
        if self.result is None:
            raise ValueError("No trained model. Run train() first.")

        output = self.result.model_path

        if format == "gguf":
            logger.warning(
                "GGUF export requires llama.cpp's convert-hf-to-gguf.py script, "
                "which must be run externally. Example:\n"
                f"  python convert-hf-to-gguf.py {self.result.model_path}"
                + (f" --outtype {quantize}" if quantize else "")
                + "\n"
                f"Returning safetensors model path: {output}"
            )
            # Return the safetensors path — no fake GGUF path
        else:
            logger.info(f"Model exported to: {output}")

        return output

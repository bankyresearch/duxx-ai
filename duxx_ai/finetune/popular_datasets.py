"""Popular HuggingFace datasets for fine-tuning — curated registry.

Pre-configured registry of 25+ popular datasets across categories:
instruction, chat, code, math, reasoning, preference, and domain-specific.
Each entry includes the HF identifier, description, sample count, format,
and recommended use case.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PopularDataset(BaseModel):
    """A curated HuggingFace dataset entry."""
    hf_id: str
    name: str
    category: str
    description: str
    samples: str          # e.g. "52K", "1.5M"
    format: str           # chat, instruction, text, preference
    license: str = ""
    subset: str = ""      # HF subset name if needed
    split: str = "train"
    recommended_for: str = ""  # e.g. "general", "code", "math"


POPULAR_DATASETS: list[PopularDataset] = [
    # ── Instruction Following ──
    PopularDataset(
        hf_id="tatsu-lab/alpaca", name="Stanford Alpaca", category="Instruction",
        description="52K instruction-following samples generated from GPT-3.5. The original instruction-tuning dataset.",
        samples="52K", format="instruction", license="cc-by-nc-4.0",
        recommended_for="General instruction following",
    ),
    PopularDataset(
        hf_id="databricks/databricks-dolly-15k", name="Dolly 15K", category="Instruction",
        description="15K human-written instruction/response pairs across 7 categories by Databricks employees.",
        samples="15K", format="instruction", license="cc-by-sa-3.0",
        recommended_for="General instruction following",
    ),
    PopularDataset(
        hf_id="yahma/alpaca-cleaned", name="Alpaca Cleaned", category="Instruction",
        description="Cleaned version of Stanford Alpaca with fixed formatting and removed duplicates.",
        samples="52K", format="instruction", license="cc-by-nc-4.0",
        recommended_for="General instruction following",
    ),
    PopularDataset(
        hf_id="WizardLMTeam/WizardLM_evol_instruct_V2_196k", name="WizardLM Evol Instruct", category="Instruction",
        description="196K evolved instruction pairs with increasing complexity for enhanced reasoning.",
        samples="196K", format="instruction", license="apache-2.0",
        recommended_for="Complex reasoning",
    ),

    # ── Chat / Conversational ──
    PopularDataset(
        hf_id="OpenAssistant/oasst2", name="OpenAssistant OASST2", category="Chat",
        description="66K multi-turn conversation trees collected from human volunteers. High quality chat data.",
        samples="66K", format="chat", license="apache-2.0",
        recommended_for="Multi-turn chat agents",
    ),
    PopularDataset(
        hf_id="timdettmers/openassistant-guanaco", name="Guanaco", category="Chat",
        description="10K curated subset of OpenAssistant for efficient fine-tuning.",
        samples="10K", format="chat", license="apache-2.0",
        recommended_for="Quick chat fine-tuning",
    ),
    PopularDataset(
        hf_id="HuggingFaceH4/ultrachat_200k", name="UltraChat 200K", category="Chat",
        description="200K multi-turn dialogues covering diverse topics. Filtered for quality.",
        samples="200K", format="chat", license="mit",
        recommended_for="General chat agents",
    ),
    PopularDataset(
        hf_id="teknium/OpenHermes-2.5", name="OpenHermes 2.5", category="Chat",
        description="1M+ diverse chat samples from multiple sources, heavily filtered for quality.",
        samples="1M+", format="chat", license="apache-2.0",
        recommended_for="High-quality chat",
    ),
    PopularDataset(
        hf_id="argilla/distilabel-capybara-dpo-7k-binarized", name="Capybara DPO", category="Chat",
        description="7K multi-turn conversations with preference labels for DPO training.",
        samples="7K", format="preference", license="apache-2.0",
        recommended_for="RLHF / DPO training",
    ),

    # ── Code ──
    PopularDataset(
        hf_id="sahil2801/CodeAlpaca-20k", name="Code Alpaca 20K", category="Code",
        description="20K code instruction/response pairs covering Python, JavaScript, Java, and more.",
        samples="20K", format="instruction", license="apache-2.0",
        recommended_for="Code generation agents",
    ),
    PopularDataset(
        hf_id="m-a-p/CodeFeedback-Filtered-Instruction", name="CodeFeedback", category="Code",
        description="66K code instructions with detailed feedback and multi-turn refinement.",
        samples="66K", format="instruction", license="apache-2.0",
        recommended_for="Code review and debugging",
    ),
    PopularDataset(
        hf_id="bigcode/the-stack-dedup", name="The Stack (Dedup)", category="Code",
        description="6.4TB of permissively licensed source code in 358 programming languages.",
        samples="6.4TB", format="text", license="various",
        subset="data/python", recommended_for="Code pretraining",
    ),

    # ── Math & Reasoning ──
    PopularDataset(
        hf_id="openai/gsm8k", name="GSM8K", category="Math",
        description="8.5K grade school math problems with step-by-step solutions.",
        samples="8.5K", format="instruction", license="mit",
        recommended_for="Math reasoning",
    ),
    PopularDataset(
        hf_id="microsoft/orca-math-word-problems-200k", name="Orca Math 200K", category="Math",
        description="200K math word problems with detailed solutions generated by GPT-4.",
        samples="200K", format="instruction", license="mit",
        recommended_for="Mathematical problem solving",
    ),
    PopularDataset(
        hf_id="TIGER-Lab/MathInstruct", name="MathInstruct", category="Math",
        description="262K math instruction samples from 13 datasets covering arithmetic to calculus.",
        samples="262K", format="instruction", license="mit",
        recommended_for="Advanced math reasoning",
    ),
    PopularDataset(
        hf_id="camel-ai/math", name="CAMEL Math", category="Math",
        description="50K math conversation pairs generated via role-playing between AI agents.",
        samples="50K", format="chat", license="cc-by-nc-4.0",
        recommended_for="Math tutoring agents",
    ),

    # ── Domain-Specific ──
    PopularDataset(
        hf_id="lavita/medical-qa-shared-task-v1-half", name="Medical QA", category="Medical",
        description="Medical question-answering dataset for healthcare AI applications.",
        samples="10K", format="instruction", license="apache-2.0",
        recommended_for="Medical AI agents",
    ),
    PopularDataset(
        hf_id="FinGPT/fingpt-sentiment-train", name="FinGPT Sentiment", category="Finance",
        description="Financial sentiment analysis dataset from news headlines and social media.",
        samples="76K", format="instruction", license="apache-2.0",
        recommended_for="Financial analysis agents",
    ),
    PopularDataset(
        hf_id="nguha/legalbench", name="LegalBench", category="Legal",
        description="Legal reasoning benchmark with 162 tasks covering legal analysis.",
        samples="20K+", format="instruction", license="cc-by-4.0",
        recommended_for="Legal AI agents",
    ),

    # ── Preference / RLHF ──
    PopularDataset(
        hf_id="Anthropic/hh-rlhf", name="Anthropic HH-RLHF", category="Preference",
        description="170K human preference comparisons for helpful and harmless AI training.",
        samples="170K", format="preference", license="mit",
        recommended_for="Safety alignment / RLHF",
    ),
    PopularDataset(
        hf_id="Intel/orca_dpo_pairs", name="Orca DPO Pairs", category="Preference",
        description="12K preference pairs for Direct Preference Optimization training.",
        samples="12K", format="preference", license="apache-2.0",
        recommended_for="DPO fine-tuning",
    ),

    # ── Multilingual ──
    PopularDataset(
        hf_id="uonlp/CulturaX", name="CulturaX", category="Multilingual",
        description="6.3T tokens across 167 languages. Cleaned web text for multilingual pretraining.",
        samples="6.3T tokens", format="text", license="cc-by-sa-4.0",
        recommended_for="Multilingual models",
    ),

    # ── Function Calling / Tool Use ──
    PopularDataset(
        hf_id="glaiveai/glaive-function-calling-v2", name="Glaive Function Calling", category="Tool Use",
        description="113K function calling examples with diverse API schemas and multi-turn conversations.",
        samples="113K", format="chat", license="apache-2.0",
        recommended_for="Tool-using agents",
    ),
    PopularDataset(
        hf_id="Salesforce/xlam-function-calling-60k", name="xLAM Function Calling", category="Tool Use",
        description="60K function calling samples from Salesforce for agent tool use training.",
        samples="60K", format="chat", license="cc-by-4.0",
        recommended_for="Agent tool calling",
    ),

    # ── Evaluation / Testing ──
    PopularDataset(
        hf_id="cais/mmlu", name="MMLU", category="Evaluation",
        description="14K multiple-choice questions across 57 subjects for testing model knowledge.",
        samples="14K", format="instruction", license="mit", split="test",
        recommended_for="Model evaluation benchmark",
    ),
    PopularDataset(
        hf_id="tatsu-lab/alpaca_eval", name="AlpacaEval", category="Evaluation",
        description="805 instructions for automated evaluation of instruction-following models.",
        samples="805", format="instruction", license="apache-2.0",
        recommended_for="Model evaluation",
    ),
]


def list_popular_datasets(category: str | None = None) -> list[dict[str, Any]]:
    """List popular datasets, optionally filtered by category."""
    result = []
    for ds in POPULAR_DATASETS:
        if category and ds.category.lower() != category.lower():
            continue
        result.append(ds.model_dump())
    return result


def get_popular_dataset(hf_id: str) -> PopularDataset | None:
    """Get a popular dataset by its HuggingFace ID."""
    for ds in POPULAR_DATASETS:
        if ds.hf_id == hf_id:
            return ds
    return None


def list_categories() -> list[str]:
    """List all available dataset categories."""
    return sorted(set(ds.category for ds in POPULAR_DATASETS))

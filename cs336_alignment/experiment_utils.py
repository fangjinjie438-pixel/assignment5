"""Shared dataset, prompt, evaluation, and logging helpers for A5 experiments.

Inputs are GSM8K JSONL files and generated response strings.  Outputs are
typed examples, formatted prompts, per-response reward records, aggregate
metrics, and append-only JSONL logs that can be plotted across random seeds.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.grpo import get_response_log_probs, tokenize_prompt_and_output
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


RewardFn = Callable[[str, str], dict[str, float]]
PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"
BUILTIN_PROMPTS = {
    "question_only": PROMPT_DIRECTORY / "question_only.prompt",
    "r1_zero": PROMPT_DIRECTORY / "r1_zero.prompt",
    "r1_zero_three_shot": PROMPT_DIRECTORY / "r1_zero_three_shot_gsm8k.prompt",
}


@dataclass(frozen=True)
class GSM8KExample:
    """One GSM8K question and its extracted final answer."""

    question: str
    ground_truth: str


@dataclass(frozen=True)
class PromptSpec:
    """Resolved prompt template plus its compatible grader and stop strings."""

    name: str
    template: str
    reward_fn: RewardFn
    stop: tuple[str, ...]

    def format(self, question: str) -> str:
        """Insert a GSM8K question into the template."""
        return self.template.format(question=question)


def extract_gsm8k_ground_truth(answer: str) -> str:
    """Extract the final answer following GSM8K's last ``####`` delimiter."""
    if "####" not in answer:
        raise ValueError("GSM8K answer does not contain the required '####' delimiter")
    ground_truth = answer.rsplit("####", maxsplit=1)[1].strip()
    if not ground_truth:
        raise ValueError("GSM8K ground truth is empty")
    return ground_truth


def load_gsm8k(path: str | Path, limit: int | None = None) -> list[GSM8KExample]:
    """Load GSM8K JSONL data.

    Args:
        path: JSONL file containing ``question`` and ``answer`` fields.
        limit: Optional maximum number of records.

    Returns:
        Parsed examples whose ground truths contain only the final answers.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")
    examples: list[GSM8KExample] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(
                    GSM8KExample(
                        question=record["question"],
                        ground_truth=extract_gsm8k_ground_truth(record["answer"]),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"missing field on line {line_number} of {path}") from exc
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"no GSM8K examples found in {path}")
    return examples


def load_prompt_spec(name_or_path: str) -> PromptSpec:
    """Resolve a built-in prompt name or a custom prompt-template path."""
    if name_or_path in BUILTIN_PROMPTS:
        name = name_or_path
        path = BUILTIN_PROMPTS[name]
    else:
        path = Path(name_or_path)
        name = path.stem
    template = path.read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError(f"prompt template {path} must contain '{{question}}'")

    # Custom prompts default to the tagged R1-Zero contract.  A custom
    # question-only prompt can opt into its boxed-answer grader via its name.
    is_question_only = name == "question_only"
    reward_fn = question_only_reward_fn if is_question_only else r1_zero_reward_fn
    stop = () if is_question_only else ("</answer>",)
    return PromptSpec(name=name, template=template, reward_fn=reward_fn, stop=stop)


def sampling_parameters(
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    seed: int,
    prompt_spec: PromptSpec,
) -> dict[str, Any]:
    """Build assignment-compliant vLLM sampling parameters."""
    if temperature < 0 or not 0 < top_p <= 1 or max_tokens <= 0 or n <= 0:
        raise ValueError("invalid sampling parameters")
    parameters: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
    }
    if prompt_spec.stop:
        parameters["stop"] = list(prompt_spec.stop)
        parameters["include_stop_str_in_output"] = True
    return parameters


def evaluate_completions(
    examples: Sequence[GSM8KExample],
    completions: Sequence[VLLMCompletion],
    prompt_spec: PromptSpec,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Grade one completion per example and return records plus mean metrics."""
    if len(examples) != len(completions):
        raise ValueError(
            f"expected one completion per example, got {len(examples)} and {len(completions)}"
        )

    records: list[dict[str, Any]] = []
    category_counts = {
        "correct_and_formatted": 0,
        "formatted_but_incorrect": 0,
        "unformatted_and_incorrect": 0,
    }
    for example, completion in zip(examples, completions, strict=True):
        reward = prompt_spec.reward_fn(completion.text, example.ground_truth)
        if reward["format_reward"] == 1 and reward["answer_reward"] == 1:
            category = "correct_and_formatted"
        elif reward["format_reward"] == 1:
            category = "formatted_but_incorrect"
        else:
            category = "unformatted_and_incorrect"
        category_counts[category] += 1
        records.append(
            {
                "question": example.question,
                "ground_truth": example.ground_truth,
                "prompt": prompt_spec.format(example.question),
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "response_length": len(completion.token_ids),
                "category": category,
                **reward,
            }
        )

    summary: dict[str, float | int] = {
        "num_examples": len(records),
        **category_counts,
        "mean_reward": fmean(record["reward"] for record in records),
        "mean_format_reward": fmean(record["format_reward"] for record in records),
        "mean_answer_reward": fmean(record["answer_reward"] for record in records),
        "mean_response_length": fmean(record["response_length"] for record in records),
    }
    return records, summary


def evaluate_policy(
    server: VLLMServer,
    examples: Sequence[GSM8KExample],
    prompt_spec: PromptSpec,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    batch_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Generate and grade one response for every example."""
    prompts = [prompt_spec.format(example.question) for example in examples]
    completions = server.generate_completions(
        prompts,
        sampling_parameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
            seed=seed,
            prompt_spec=prompt_spec,
        ),
        batch_size=batch_size,
    )
    return evaluate_completions(examples, completions, prompt_spec)


def compute_old_log_probs(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    responses: list[str],
    microbatch_size: int,
) -> torch.Tensor:
    """Score rollouts under a frozen policy and return CPU log probabilities.

    The full batch is tokenized before splitting so all returned rows share a
    sequence length.  Output shape is ``(len(prompts), max_length - 1)``.
    """
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    chunks: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), microbatch_size):
                input_ids = tokenized["input_ids"][start : start + microbatch_size].to(device)
                labels = tokenized["labels"][start : start + microbatch_size].to(device)
                chunks.append(
                    get_response_log_probs(model, input_ids, labels)["log_probs"].cpu()
                )
    finally:
        model.train(was_training)
    return torch.cat(chunks, dim=0)


def numeric_metadata_mean(
    metadata_items: Sequence[Mapping[str, torch.Tensor | float]],
) -> dict[str, float]:
    """Average scalar train-step metadata across optimizer minibatches."""
    if not metadata_items:
        raise ValueError("metadata_items must not be empty")
    keys = set.intersection(*(set(item) for item in metadata_items))
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = []
        for item in metadata_items:
            value = item[key]
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().item()
            values.append(float(value))
        if len(values) == len(metadata_items):
            result[key] = fmean(values)
    return result


def write_json(path: str | Path, value: Any) -> None:
    """Write a readable UTF-8 JSON file, creating parent directories."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records as UTF-8 JSON Lines, replacing an existing file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one record to an experiment JSONL log."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as output:
        output.write(json.dumps(dict(record), ensure_ascii=False) + "\n")

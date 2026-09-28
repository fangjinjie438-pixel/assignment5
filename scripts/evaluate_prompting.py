#!/usr/bin/env python3
"""Evaluate the three required prompt baselines on GSM8K.

Input: model/data paths, prompt names, generation settings, and GPU selection.
Output: per-prompt response JSONL files, summary metrics, and two ten-example
audit files for manually checking parser false negatives in categories 2/3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_alignment.experiment_utils import (
    evaluate_policy,
    load_gsm8k,
    load_prompt_spec,
    write_json,
    write_jsonl,
)
from cs336_alignment.vllm_utils import VLLMServer


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--data", default="data/gsm8k/test.jsonl")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["question_only", "r1_zero", "r1_zero_three_shot"],
    )
    parser.add_argument("--num-examples", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="experiments/prompting_baselines")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    examples = load_gsm8k(args.data, args.num_examples)
    output_dir = Path(args.output_dir)
    summaries = {}
    server = VLLMServer(model_id=args.model, gpu=args.gpu, seed=args.seed)
    server.start()
    try:
        for prompt_index, prompt_name in enumerate(args.prompts):
            prompt_spec = load_prompt_spec(prompt_name)
            records, summary = evaluate_policy(
                server,
                examples,
                prompt_spec,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=args.seed + prompt_index,
                batch_size=args.batch_size,
            )
            summaries[prompt_spec.name] = summary
            write_jsonl(output_dir / f"{prompt_spec.name}_responses.jsonl", records)

            # The handout explicitly asks for manual inspection of at least ten
            # category-2 and category-3 responses.  These files make that audit
            # deterministic and keep human judgments separate from grader data.
            formatted_wrong = [
                record for record in records if record["category"] == "formatted_but_incorrect"
            ][:10]
            unformatted = [
                record for record in records if record["category"] == "unformatted_and_incorrect"
            ][:10]
            write_jsonl(
                output_dir / f"{prompt_spec.name}_audit_category_2.jsonl",
                formatted_wrong,
            )
            write_jsonl(
                output_dir / f"{prompt_spec.name}_audit_category_3.jsonl",
                unformatted,
            )
            print(prompt_spec.name, summary, flush=True)
    finally:
        server.stop()
    write_json(output_dir / "summary.json", summaries)


if __name__ == "__main__":
    main()
    
    # from cs336_alignment.vllm_utils import VLLMServer


    # def generate_responses(
    #     model_id: str,
    #     prompts: list[str],
    #     stop: list[str] | None = None,
    # ) -> list[str]:
        
    #     server = VLLMServer(
    #         model_id=model_id,
    #         gpu=0,
    #         port=8000,
    #         seed=0,
    #     )
    #     try:
    #         server.start()
    #         completions = server.generate_completions(
    #             prompts=prompts,
    #             sampling_params={
    #                 "temperature": 1.0,
    #                 "top_p": 1.0,
    #                 "max_tokens": 512,
    #                 "n": 1,
    #                 "seed": 0,
    #                 "stop": stop,
    #                 "include_stop_str_in_output": True
    #             },
    #             batch_size=64,
    #         )
    #         return [completion.text for completion in completions]
    #     finally:
    #         # 即使生成过程报错，也要释放 GPU 和端口。
    #         server.stop()

    # prompts = []
    # stop = []

    # completions = generate_responses(model_id= "allenai/OLMo-2-0425-1B", prompts= prompts, stop= stop)

    
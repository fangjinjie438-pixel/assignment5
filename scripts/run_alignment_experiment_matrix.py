#!/usr/bin/env python3
"""Build or sequentially execute the on-policy assignment experiment matrix.

Every generated training command uses scripts/my_grpo_alignment.py, which is a
copy of scripts/my_grpo_clean.py with the alignment experiment extensions.
Commands are printed by default; --execute runs them one at a time.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


PROMPT_FILES = {
    "r1_zero": "r1_zero.prompt",
    "question_only": "question_only.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}
SUITES = ("standard", "learning_rate", "prompt_ablation", "variants", "all")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="learning_rate")
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--learning-rates", default="3e-6,3e-5")
    parser.add_argument("--base-learning-rate", type=float, default=1e-5)
    parser.add_argument("--output-root", default="experiments/alignment_new")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="training arguments after '--' forwarded to every run")
    return parser


def command(algorithm: str, prompt: str, seed: int, learning_rate: float,
            output_dir: str, extra_args: list[str]) -> list[str]:
    result = [
        sys.executable, "scripts/my_grpo_alignment.py",
        "--algorithm", algorithm,
        "--prompt_path", f"cs336_alignment/prompts/{PROMPT_FILES[prompt]}",
        "--seed", str(seed),
        "--learning_rate", str(learning_rate),
        "--output_dir", output_dir,
    ]
    return result + (extra_args[1:] if extra_args[:1] == ["--"] else extra_args)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    seeds = [int(seed) for seed in args.seeds.split(",")]
    rates = [float(rate) for rate in args.learning_rates.split(",")]
    if not seeds or not rates or args.base_learning_rate <= 0 or any(rate <= 0 for rate in rates):
        raise ValueError("seeds must be nonempty and learning rates must be positive")
    suites = SUITES[:-1] if args.suite == "all" else (args.suite,)
    commands: list[list[str]] = []
    root = args.output_root
    if "standard" in suites:
        for seed in seeds:
            commands.append(command("standard", "r1_zero", seed, args.base_learning_rate,
                                    f"{root}/standard", args.extra_args))
    if "learning_rate" in suites:
        for rate in rates:
            for seed in seeds:
                commands.append(command("standard", "r1_zero", seed, rate,
                                        f"{root}/learning_rate/lr_{rate:g}", args.extra_args))
    if "prompt_ablation" in suites:
        for prompt in ("question_only", "r1_zero_three_shot"):
            for seed in seeds:
                commands.append(command("standard", prompt, seed, args.base_learning_rate,
                                        f"{root}/prompt_ablation/{prompt}", args.extra_args))
    if "variants" in suites:
        for algorithm in ("grpo_constant", "dr_grpo", "rft", "maxrl"):
            for seed in seeds:
                commands.append(command(algorithm, "r1_zero", seed, args.base_learning_rate,
                                        f"{root}/variants/{algorithm}", args.extra_args))
    return commands


def main() -> None:
    args = make_parser().parse_args()
    for training_command in build_commands(args):
        print(shlex.join(training_command), flush=True)
        if args.execute:
            subprocess.run(training_command, check=True)


if __name__ == "__main__":
    main()

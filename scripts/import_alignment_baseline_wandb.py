#!/usr/bin/env python3
"""Upload the four existing my_grpo_clean.py standard runs to W&B once."""

# uv run python scripts/import_alignment_baseline_wandb.py   experiments/grpo_standard_on_policy/seed_{42,43,44,45}/metrics.jsonl   --project cs336-assignment5

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", default="standard-existing")
    args = parser.parse_args()

    import wandb

    for metrics_path in args.logs:
        config = json.loads(metrics_path.with_name("config.json").read_text(encoding="utf-8"))
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        expected_step = config["num_rollout_steps"]
        final_val = [r for r in records if r["kind"] == "val" and r["step"] == expected_step]
        if len(final_val) != 1:
            raise ValueError(f"{metrics_path}: expected one final validation at step {expected_step}")
        seed = config["seed"]
        with wandb.init(
            project=args.project,
            group=args.group,
            job_type="standard",
            name=f"standard-r1_zero-lr{config['learning_rate']:g}-seed{seed}-existing",
            config=config | {"algorithm": "standard", "prompt_name": "r1_zero", "source_metrics": str(metrics_path)},
        ) as run:
            run.define_metric("rollout_step")
            run.define_metric("train/*", step_metric="rollout_step")
            run.define_metric("val/*", step_metric="rollout_step")
            for record in records:
                kind = record["kind"]
                run.log({
                    "rollout_step": record["step"],
                    **{
                        f"{kind}/{key}": value
                        for key, value in record.items()
                        if isinstance(value, (int, float)) and key not in {"step", "seed"}
                    },
                })
            run.summary["final_val_reward"] = final_val[0]["mean_reward"]


if __name__ == "__main__":
    main()

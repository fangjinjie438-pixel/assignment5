# 这里的NCCL_CUMEM_HOST_ENABLE=0是适用于GPU1和4否则会阻塞，如果是457GPU可以通过PIX互相通信可能不需要这个就可以直接运行
# NCCL_CUMEM_HOST_ENABLE=0
# uv run python scripts/my_grpo_clean.py \
#   --model_name allenai/OLMo-2-0425-1B \
#   --prompt_path cs336_alignment/prompts/r1_zero.prompt \
#   --train_data data/gsm8k/train.jsonl \
#   --val_data data/gsm8k/test.jsonl \
#   --num_rollout_steps 200 \
#   --n_train_examples 6400 \
#   --n_val_examples 1024 \
#   --rollout_batch_size 256 \
#   --train_batch_size 256 \
#   --group_size 8 \
#   --gradient_accumulation_steps 64 \
#   --learning_rate 1e-5 \
#   --max_grad_norm 1.0 \
#   --temperature 1.0 \
#   --top_p 1.0 \
#   --max_tokens 512 \
#   --eval_every 10 \
#   --log_rollouts_every 40 \
#   --output_dir experiments/grpo_standard_on_policy \
#   --seed 42 \
#   --training_device cuda:4 \
#   --inference_gpu 1


# 如果只是--eval_only，那么uv run python scripts/my_grpo_clean.py --eval_only --n_val_examples 1024 --inference_gpu 1

import torch
import argparse
import json
import random
from statistics import fmean
from dataclasses import dataclass
from pathlib import Path

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.my_grpo_alignment import grpo_train_step

ALGORITHMS = {
    "standard": ("mean", "std", "sequence"),
    "grpo_constant": ("mean", "std", "constant"),
    "dr_grpo": ("mean", "none", "constant"),
    "rft": ("none", "none", "constant"),
    "maxrl": ("mean", "mean", "constant"),
}


def prompt_name(prompt_path: str) -> str:
    name = Path(prompt_path).stem
    return "r1_zero_three_shot" if name == "r1_zero_three_shot_gsm8k" else name


def reward_fn_for_prompt(prompt_path: str):
    return question_only_reward_fn if prompt_name(prompt_path) == "question_only" else r1_zero_reward_fn


def sampling_stop(prompt_path: str) -> dict:
    if prompt_name(prompt_path) == "question_only":
        return {}
    return {"stop": ["</answer>"], "include_stop_str_in_output": True}


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    ground_truth: str

@dataclass
class RolloutBatch:
    repeated_prompts: list[str]
    responses: list[str]
    repeated_ground_truths: list[str]
    response_lengths: list[int]

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="standard")
    parser.add_argument("--prompt_path", default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--train_data", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data", default="data/gsm8k/test.jsonl")
    # 这个参数表示的是train和val的example数目
    parser.add_argument("--n_train_examples", type=int, default=6400)
    parser.add_argument("--n_val_examples", type=int, default=1024)
    
    # rollout_batch_size 是每轮采样得到的回答总数，train_batch_size 是一次参数更新使用的回答数；
    # 两者都按回答数计算，不是按题目数。标准 on-policy 配置中二者都是 256：对 32 道题各采样 8 个回答，然后用这 256 个回答做一次更新。
    # 如果 train_batch_size 更小，就要把同一轮 rollout 分成多个训练 batch、执行多次更新；后面的更新使用的便是更新前策略采样的数据，不再是严格的单次更新 on-policy 设置。

    # rollout_step总步数，每一步代表一次vLLM推理
    parser.add_argument("--num_rollout_steps", type=int, default=200)
    # rollout_batch_size表示：一次rollout的生成的回答总数
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    # train_batch_size一次更新用到的回答数
    parser.add_argument("--train_batch_size", type=int, default=256)

    # 每个问题对应gorup_size个回答
    parser.add_argument("--group_size", type=int, default=8)

    # 正常整个batch_size进行计算loss然后更新，但是由于显存不足需要分成gradient_accumulation_steps进行更新
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)

    # 初始化优化器的learning_rate
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    # clip模型参数的max_grad_norm
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 这三个参数是vLLM生成的参数
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)

    # generation_batch_size是在vLLM推理生成的response个数(但是最后要乘上n)
    parser.add_argument("--generation_batch_size", type=int, default=64)
    # eval的频率
    parser.add_argument("--eval_every", type=int, default=10)
    # 记录rollout规矩的频率
    parser.add_argument("--log_rollouts_every", type=int, default=40)
    
    # 记录metrics，config，validation_step_...的根目录
    parser.add_argument("--output_dir", default="experiments/my_grpo")

    # TODO: wandb_project这个参数的作用在哪？如何使用wandb？
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    # TODO: seed到底作用在哪里？
    parser.add_argument("--seed", type=int, default=0)
    # training_device是模型训练的设备，inference_gpu是vLLM推理的设备
    parser.add_argument("--training_device", default="cuda:0")
    parser.add_argument("--inference_gpu", type=int, default=1)
    parser.add_argument("--vllm_port", type=int, default=None)

    # 是否只是进行step0的eval_only
    parser.add_argument("--eval_only", action="store_true")
    return parser

# 参数合法性检查
def validate_args(args: argparse.Namespace) -> None:
    # TODO: rollout_batch_size如果是off_policy和train_batch_size之间的关系是什么？
    if args.rollout_batch_size != args.train_batch_size:
        raise ValueError(
            "标准 on-policy 实验中，每个 rollout 应恰好使用一次；"
            "因此 rollout_batch_size 应等于 train_batch_size"
        )
    if args.rollout_batch_size % args.group_size != 0:
        raise ValueError("rollout_batch_size 必须能被 group_size 整除")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size 必须能被 gradient_accumulation_steps 整除")
    if args.n_train_examples < (args.rollout_batch_size // args.group_size):
        raise ValueError("训练集不足以构造一个 rollout batch")
    if args.eval_every <= 0 or args.log_rollouts_every <= 0:
        raise ValueError("日志间隔必须为正数")
    if args.learning_rate <= 0 or args.max_tokens <= 0 or args.num_rollout_steps <= 0:
        raise ValueError("learning_rate、max_tokens 和 num_rollout_steps 必须为正数")
    if args.vllm_port is not None and not 1 <= args.vllm_port <= 65535:
        raise ValueError("vllm_port 必须在 1 到 65535 之间")

# 从"####"后提取最终答案:
def extract_gsm8k_ground_truth(answer: str) -> str:
    # 由"####"作为分界线，分别为三部分str
    prefix, separator, ground_truth = answer.rpartition("####")
    if not separator:
        raise ValueError("GSM8K answer 中没有找到 '####' 分隔符")
    ground_truth = ground_truth.strip()
    if not ground_truth:
        raise ValueError("GSM8K ground truth 不能为空")
    return ground_truth

# 加载jsonl:提取数据中的问题和答案
def load_gsm8k(path: str | Path) -> list[GSM8KExample]:
    path = Path(path)
    examples: list[GSM8KExample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "question" not in record:
                raise ValueError(f"{path} 第 {line_number} 行缺少 question")
            if "answer" not in record:
                raise ValueError(f"{path} 第 {line_number} 行缺少 answer")
            examples.append(
                GSM8KExample(
                    question=record["question"],
                    ground_truth=extract_gsm8k_ground_truth(record["answer"])
                )
            )
    if not examples:
        raise ValueError(f"{path} 中没有找到样本")
    return examples

# load后format填充对应的answer
def load_prompt(path: str | Path) -> str:
    path = Path(path)
    template = path.read_text(encoding="utf-8")
    if "{question}" not in template:
        raise ValueError(f"prompt 模板 {path} 中没有 {{question}} 占位符")
    return template

def format_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.format(question=question)

# 默认配置：
# 每步 32 道题
# 每道题 8 个 response
# 总计 256 个 response

# 200 步 × 32 道题 = 6400 道题
# 从rollout_step开始抽取prompt_batch_size个prompt
def select_train_batch(
    train_examples: list[GSM8KExample],
    rollout_step: int,
    args: argparse.Namespace,
) -> list[GSM8KExample]:
    prompt_batch_size = args.rollout_batch_size // args.group_size
    start = rollout_step * prompt_batch_size
    return [
        train_examples[(start + index) % len(train_examples)]
        for index in range(prompt_batch_size)
    ]

def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def evaluate(
    server: VLLMServer,
    examples: list[GSM8KExample],
    prompt_template: str,
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, float]]:
    prompts = [
        format_prompt(prompt_template, example.question)
        for example in examples
    ]
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": 1,
            "seed": args.seed + 100_000,
            **sampling_stop(args.prompt_path),
        },
        # 这里的generation_batch_size是prompts中选取作为一批生成的prompt(这里的prompt是填充后问题的模板)
        batch_size=args.generation_batch_size,
    )
    if len(completions) != len(examples):
        raise RuntimeError(
            f"期望 {len(examples)} 个验证回答，"
            f"实际得到 {len(completions)} 个"
        )
    records = []
    for example, prompt, completion in zip(
        examples, prompts, completions, strict=True
    ):
        reward = reward_fn_for_prompt(args.prompt_path)(
            completion.text,
            example.ground_truth,
        )
        records.append({
            "question": example.question,
            "ground_truth": example.ground_truth,
            "prompt": prompt,
            "response": completion.text,
            "response_length": len(completion.token_ids),
            "finish_reason": completion.finish_reason,
            **reward,
        })
    metrics = {
        "mean_reward": fmean(x["reward"] for x in records),
        "mean_format_reward": fmean(
            x["format_reward"] for x in records
        ),
        "mean_response_length": fmean(
            x["response_length"] for x in records
        ),
        # 补充
        "mean_answer_reward": fmean(
            x["answer_reward"] for x in records
        ),
    }
    return records, metrics

# 生成函数
def generate_training_rollouts(
    server: VLLMServer,
    examples: list[GSM8KExample],
    prompt_template: str,
    rollout_step: int,
    args: argparse.Namespace,
) -> RolloutBatch:
    prompts = [
        format_prompt(prompt_template, x.question)
        for x in examples
    ]
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            # 对于同一个问题重复n次回答
            "n": args.group_size,
            "seed": args.seed + rollout_step + 1,
            **sampling_stop(args.prompt_path),
        },
        # 这里的generation_batch_size指的是一次推理生成这么多问题的response
        batch_size=args.generation_batch_size,
    )
    expected = len(examples) * args.group_size
    if len(completions) != expected:
        raise RuntimeError(
            f"期望 {expected} 个 rollout，"
            f"实际得到 {len(completions)} 个"
        )
    repeated_prompts = [
        prompt
        for prompt in prompts
        for _ in range(args.group_size)
    ]
    repeated_ground_truths = [
        example.ground_truth
        for example in examples
        for _ in range(args.group_size)
    ]
    return RolloutBatch(
        repeated_prompts=repeated_prompts,
        responses=[x.text for x in completions],
        repeated_ground_truths=repeated_ground_truths,
        response_lengths=[
            len(x.token_ids) for x in completions
        ],
    )

# 检查分组是否正确
def check_rollout_batch(
    batch: RolloutBatch,
    group_size: int,
) -> None:
    size = len(batch.responses)
    if not (
        len(batch.repeated_prompts)
        == len(batch.repeated_ground_truths)
        == len(batch.response_lengths)
        == size
    ):
        raise ValueError("rollout batch 各字段长度不一致")
    if size % group_size != 0:
        raise ValueError("rollout 数量不能被 group_size 整除")
    for start in range(0, size, group_size):
        stop = start + group_size
        if len(set(batch.repeated_prompts[start:stop])) != 1:
            raise ValueError("同一 group 中出现了不同 prompt")
        if len(set(batch.repeated_ground_truths[start:stop])) != 1:
            raise ValueError("同一 group 中出现了不同答案")

# 观察初始 rollout 是否有训练信号
def inspect_rollout_rewards(
    batch: RolloutBatch,
    group_size: int,
    reward_fn,
) -> dict[str, float]:
    rewards = torch.tensor([
        reward_fn(response, ground_truth)["reward"]
        for response, ground_truth in zip(
            batch.responses,
            batch.repeated_ground_truths,
            strict=True,
        )
    ]).reshape(-1, group_size)
    successes = rewards.sum(dim=1)

    # 其中 mixed_group_fraction 很重要：
    # - 全组 reward 都是 0：advantage 全为 0；
    # - 全组 reward 都是 1：advantage也全为 0；
    # - 同组中有对有错：标准 GRPO 才会产生有效梯度。
    return {
        "mean_reward": rewards.mean().item(),
        "pass_at_group_size": (
            successes > 0
        ).float().mean().item(),
        "mixed_group_fraction": (
            (successes > 0) & (successes < group_size)
        ).float().mean().item(),
        "mean_response_length": (
            sum(batch.response_lengths)
            / len(batch.response_lengths)
        ),
    }

def scalarize(metadata: dict) -> dict[str, float]:
    result = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[key] = value.detach().item()
        elif isinstance(value, (int, float)):
            result[key] = float(value)
    return result

def perform_grpo_update(
    policy: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    rollout_batch: RolloutBatch,
    args: argparse.Namespace,
) -> dict[str, float]:
    loss, metadata = grpo_train_step(
        model=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        max_grad_norm=args.max_grad_norm,
        reward_fn=reward_fn_for_prompt(args.prompt_path),
        repeated_prompts=rollout_batch.repeated_prompts,
        rollout_responses=rollout_batch.responses,
        repeated_ground_truths=(
            rollout_batch.repeated_ground_truths
        ),
        group_size=args.group_size,

        baseline=ALGORITHMS[args.algorithm][0],
        advantage_eps=1e-6,
        advantage_normalizer=ALGORITHMS[args.algorithm][1],
        importance_reweighting_method="none",
        old_log_probs=None,
        cliprange=None,
        loss_normalization=ALGORITHMS[args.algorithm][2],
        normalization_constant=(
            args.train_batch_size * args.max_tokens
            if ALGORITHMS[args.algorithm][2] == "constant" else None
        ),
    )
    metrics = scalarize(metadata)
    metrics["loss"] = loss.detach().item()
    metrics["mean_response_length"] = (
        sum(rollout_batch.response_lengths)
        / len(rollout_batch.response_lengths)
    )
    diagnostics = inspect_rollout_rewards(
        rollout_batch,
        args.group_size,
        reward_fn_for_prompt(args.prompt_path),
    )
    metrics.update({
        "pass_at_group_size": diagnostics[
            "pass_at_group_size"
        ],
        "mixed_group_fraction": diagnostics[
            "mixed_group_fraction"
        ],
    })
    return metrics

import math

def check_train_metrics(metrics: dict[str, float]) -> None:
    required = [
        "loss",
        "gradient_norm",
        "token_entropy",
        "mean_reward",
        "mean_format_reward",
    ]
    for key in required:
        if key not in metrics:
            raise RuntimeError(f"训练指标缺少 {key}")
        if not math.isfinite(metrics[key]):
            raise RuntimeError(
                f"{key} 出现非有限值：{metrics[key]}"
            )

# 通用json工具
def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# 初始化运行目录和 W&B
def initialize_run(args):
    run_dir = Path(args.output_dir) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    check_run_dir_available(args)
    write_json(run_dir / "config.json", vars(args))
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.algorithm}-{prompt_name(args.prompt_path)}-lr{args.learning_rate:g}-seed{args.seed}",
            group=args.wandb_group or Path(args.output_dir).name,
            job_type=args.algorithm,
            config=vars(args) | {"prompt_name": prompt_name(args.prompt_path)},
        )
        wandb_run.define_metric("rollout_step")
        wandb_run.define_metric("train/*", step_metric="rollout_step")
        wandb_run.define_metric("val/*", step_metric="rollout_step")
    return run_dir, metrics_path, wandb_run


def check_run_dir_available(args) -> None:
    metrics_path = Path(args.output_dir) / f"seed_{args.seed}" / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"{metrics_path} 已存在，拒绝混合两次实验")

# 记录指标
def log_metrics(
    record: dict,
    metrics_path: Path,
    wandb_run,
) -> None:
    append_jsonl(metrics_path, record)
    if wandb_run is not None:
        kind = record["kind"]
        wandb_run.log({
            "rollout_step": record["step"],
            **{
                f"{kind}/{key}": value
                for key, value in record.items()
                if isinstance(value, (int, float))
                and key not in {"step", "seed"}
            },
        })

# 保存训练rollout
def save_rollouts(
    path: Path,
    batch: RolloutBatch,
    args: argparse.Namespace,
) -> None:
    records = []
    for prompt, response, ground_truth, length in zip(
        batch.repeated_prompts,
        batch.responses,
        batch.repeated_ground_truths,
        batch.response_lengths,
        strict=True,
    ):
        reward = reward_fn_for_prompt(args.prompt_path)(response, ground_truth)
        records.append({
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "response_length": length,
            **reward,
        })
    write_jsonl(path, records)

def evaluate_and_log(
    step: int,
    server: VLLMServer,
    examples: list[GSM8KExample],
    prompt_template: str,
    args,
    run_dir: Path,
    metrics_path: Path,
    wandb_run,
) -> dict[str, float]:
    # records表示的是基本的问题回答的记录，metrics表示的是reward各种指标
    records, metrics = evaluate(
        server=server,
        examples=examples,
        prompt_template=prompt_template,
        args=args,
    )
    write_jsonl(
        run_dir / f"validation_step_{step:04d}.jsonl",
        records,
    )
    # 记录指标到wandb和metrics_path中
    log_metrics(
        {
            "kind": "val",
            "step": step,
            "seed": args.seed,
            "algorithm": args.algorithm,
            "prompt_name": prompt_name(args.prompt_path),
            "learning_rate": args.learning_rate,
            **metrics,
        },
        metrics_path,
        wandb_run,
    )
    return metrics

def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    validate_args(args)
    check_run_dir_available(args)

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 训练集的shuffle和准备训练/验证数据
    all_train_examples = load_gsm8k(args.train_data)
    all_val_examples = load_gsm8k(args.val_data)

    train_rng = random.Random(args.seed)
    train_rng.shuffle(all_train_examples)

    train_examples = all_train_examples[:args.n_train_examples]
    val_examples = all_val_examples[:args.n_val_examples]

    if len(train_examples) < args.n_train_examples:
        raise ValueError(
            f"训练集只有 {len(train_examples)} 条，"
            f"但要求 {args.n_train_examples} 条"
        )
    if len(val_examples) < args.n_val_examples:
        raise ValueError(
            f"验证集只有 {len(val_examples)} 条，"
            f"但要求 {args.n_val_examples} 条"
        )

    # 加载并格式化prompt, format_prompt在后面的函数中有调用
    prompt_template = load_prompt(args.prompt_path)

    wandb_run = None

    # 启动vLLM
    server = VLLMServer(
        model_id=args.model_name,
        port=args.vllm_port if args.vllm_port is not None else 8000 + args.inference_gpu,
        gpu=args.inference_gpu,
        seed=args.seed,
    )
    server.start()

    # 训练时，合适的顺序是：启动 vLLM → 加载训练模型 → 初始化同步组一次 → 每次训练更新权重后，在下一次 vLLM 生成或验证前同步一次
    try:
        # 初始化运行目录和 W&B，将运行命令配置写入config文件
        run_dir, metrics_path, wandb_run = initialize_run(args)

        if args.eval_only:
            evaluate_and_log(
                step=0,
                server=server,
                examples=val_examples,
                prompt_template=prompt_template,
                args=args,
                run_dir=run_dir,
                metrics_path=metrics_path,
                wandb_run=wandb_run,
            )
            return

        # 加载policy训练模型和tokenizer
        policy, tokenizer = get_model_and_tokenizer(
            args.model_name,
            args.training_device,
        )
        policy.train()

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )

        # 让训练进程和 vLLM 建立 NCCL 通信组。只初始化通道，不传模型权重；通常在 vLLM 启动、训练模型加载后，每次启动服务调用一次
        server.init_weight_sync(args.training_device)
        # 把训练模型当前的参数传给 vLLM，让之后的生成使用新权重；代码还会暂停服务、更新权重、清除前缀缓存并恢复服务。它不传梯度或优化器状态
        server.sync_policy_weights(policy)


        # 训练前记录,step 0 步记录对应的evaluate
        evaluate_and_log(
            step=0,
            server=server,
            examples=val_examples,
            prompt_template=prompt_template,
            args=args,
            run_dir=run_dir,
            metrics_path=metrics_path,
            wandb_run=wandb_run,
        )

        for rollout_step in range(args.num_rollout_steps):
            # 默认配置：
            # 每rollout_step步 选取 32 道题
            # 每道题 8 个 response
            # 总计 256 个 response

            # 200 步 × 32 道题 = 6400 道题
            # 从rollout_step开始抽取prompt_batch_size个prompt

            # 这一步选出 32 道题
            examples = select_train_batch(
                train_examples=train_examples,
                rollout_step=rollout_step,
                args=args,
            )

            # 每次采样前同步最新 policy
            server.sync_policy_weights(policy)

            # 每道题生成 8 个回答，共 256 个 response。
            rollout_batch = generate_training_rollouts(
                server=server,
                examples=examples,
                prompt_template=prompt_template,
                rollout_step=rollout_step,
                args=args,
            )

            # 检查最终的rollout_batch是否分组合理(prompt + response)
            check_rollout_batch(
                rollout_batch,
                args.group_size,
            )

            # 内部调用一次你的 grpo_train_step，进行参数的更新
            train_metrics = perform_grpo_update(
                policy=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                rollout_batch=rollout_batch,
                args=args,
            )

            check_train_metrics(train_metrics)

            print({
                "kind": "train",
                "step": rollout_step + 1,
                **train_metrics,
            })


            step = rollout_step + 1

            log_metrics(
                {
                    "kind": "train",
                    "step": step,
                    "seed": args.seed,
                    "algorithm": args.algorithm,
                    "prompt_name": prompt_name(args.prompt_path),
                    "learning_rate": args.learning_rate,
                    **train_metrics,
                },
                metrics_path,
                wandb_run,
            )

            if step % args.log_rollouts_every == 0:
                save_rollouts(
                    run_dir / f"train_rollouts_step_{step:04d}.jsonl",
                    rollout_batch,
                    args,
                )

            if step % args.eval_every == 0 or step == args.num_rollout_steps:
                server.sync_policy_weights(policy)

                evaluate_and_log(
                    step=step,
                    server=server,
                    examples=val_examples,
                    prompt_template=prompt_template,
                    args=args,
                    run_dir=run_dir,
                    metrics_path=metrics_path,
                    wandb_run=wandb_run,
                )

        checkpoint_dir = run_dir / "final_checkpoint"
        policy.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        server.stop()

if __name__ == "__main__":
    main()
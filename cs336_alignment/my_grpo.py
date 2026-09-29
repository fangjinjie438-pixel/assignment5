from collections.abc import Callable
from typing import Literal

import torch
from cs336_alignment.checkpoint import get_model_and_tokenizer
from transformers import PreTrainedTokenizerBase, PreTrainedModel

# 这里tokenize_prompt_and_output是如何引入tokneizer的->cs336_alignment/checkpoint中引入

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    # prompt_strs和output_sts的长度都是batch_size,那么cat+padding的总长度就是seq_len+1
    # 于是最终的input_ids和labels和response_mask的shape: (batch_size, seq_len)
    # 注意response_mask是对label进行mask的，label的output部分为True，其余部分为False
    assert len(prompt_strs)==len(output_strs), "输入的prompt和输出的output个数不相等"

    if not prompt_strs:
        raise ValueError("prompt_strs and output_strs不能为空")

    # TODO:这里的add_special_tokens是什么意思？
    prompt_ids = [tokenizer.encode(prompt, add_special_tokens = False) for prompt in prompt_strs]
    output_ids = [tokenizer.encode(prompt, add_special_tokens = False) for prompt in output_strs]

    combined_ids = []
    no_padding_num = []
    input_id_num = []
    output_id_num = []  
    prompt_and_output_lens = 0

    for prompt_list, output_list in zip(prompt_ids, output_ids):
        prompt_list_len = len(prompt_list)
        output_list_len = len(output_list)

        input_id_num.append(prompt_list_len)
        output_id_num.append(output_list_len)

        plus_num = prompt_list_len + output_list_len
        prompt_and_output_lens = max(plus_num , prompt_and_output_lens)
        no_padding_num.append(plus_num)
        
        combined_ids.append(prompt_list + output_list)

    if prompt_and_output_lens < 2:
        raise ValueError(
            "prompt不应该只有一个甚至更少的token"
        )

    padding_id = tokenizer.pad_token_id
    padding_num = [prompt_and_output_lens - per_no_padding_num for per_no_padding_num in no_padding_num]
    if sum(padding_num)!=0 and padding_id is None:
        raise ValueError(
            "tokenzier中不存在padding_id"
        )
    if padding_id is None:
        padding_id = 0

    assert len(combined_ids) == len(padding_num), "padding_num和combined_ids的个数不一样"
    combined_ids_with_padding = [per_combined_ids + [padding_id] * per_padding_num for per_combined_ids, per_padding_num in zip(combined_ids, padding_num, strict=True)]

    # TODO:这里要设置为torch.long？？？？
    final_combined_tensor = torch.tensor(combined_ids_with_padding, dtype= torch.long)

    input_ids = final_combined_tensor[..., :-1]
    labels = final_combined_tensor[..., 1:]
    response_mask = torch.tensor([[False] * (x-1) + [True] * y + [False] * z for x,y,z in zip(input_id_num, output_id_num, padding_num, strict=True)], dtype=torch.bool)


    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }


# # 对于得到的input_ids 和 labels，我们可以通过如下方法得到对应的：
# input_ids = train_batch["input_ids"].to(device)
# labels = train_batch["labels"].to(device)
# logits = model(input_ids).logits


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    # 输入的input_ids和labels: (batch_size, seq_len)
    # 输出的log_probs和token_entropy: (batch_size, seq_len),
    """
    input_ids : shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method
    labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your tokenization method
    """
    # 这里的log_probs是真实label对应的log_probility；这里的token_entropy指的是对应一个token的交叉熵，∑p_i * log p_i（对应的i类别的概率为p_i，i=0,1,....,vocab_size-1）
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError("input_ids and labels must both be rank-2 tensors")
    if input_ids.shape != labels.shape:
        raise ValueError(
            f"input_ids and labels must have the same shape (got {input_ids.shape} "
            f"and {labels.shape})"
        )

    # logits: (batch_size, seq_len, vocab_size)
    # 这里必须要有.logits因为huggingface Transformers返回的不是纯Tensor，而是一个ModelOutput对象
    logits = model(input_ids).logits

    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "model logits must match labels in the batch and sequence dimensions "
            f"(got {logits.shape[:2]} and {labels.shape})"
        )


    # log_probs和probs: (batch_size, seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # output_log_probs: (batch_size, seq_len)
    output_log_probs = torch.gather(input = log_probs, dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        token_entropy = torch.sum(-probs * log_probs, dim = -1)
        return {
            "log_probs": output_log_probs,
            "token_entropy": token_entropy
        }
    return {
        "log_probs": output_log_probs
    }


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    # 输入的rollout_responses和repeated_ground_truths.shape: (rollout_batch_size = batch_size, )
    # 输出的raw_rewards.shape (rollout_batch_size = batch_size, ), reward_metadata
    """
    • reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the 
    ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    
    • rollout_responses: list[str] Rollouts from the policy. The length of this list is 
    rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    
    • repeated_ground_truths: list[str] The ground truths for the examples. The length of this 
    list is rollout_batch_size, because the ground truth for each example is repeated group_size
    times.
    
    Returns:
    • tuple[torch.Tensor, dict[str, float]].
        ‣ raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        ‣ metadata Reward statistics to log. At minimum, include the mean total and format rewards 
    over the rollout batch.
    """
    # 这两个都是list[str]，形状是: (rollout_batch_size,), rollout_batch_size = prompt数 * group_size
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have equal length "
            f"(got {len(rollout_responses)} and {len(repeated_ground_truths)})"
        )
    if not rollout_responses:
        raise ValueError("at least one rollout response is required")

    # scores: list[dict[str, float]], str: "reward", "format_reward", and "answer_reward"
    scores = [reward_fn(response, ground_truth) for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)]


    required_components = ("reward", "format_reward", "answer_reward")
    for index, score in enumerate(scores):
        missing = set(required_components).difference(score)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise KeyError(f"reward_fn result {index} is missing: {missing_names}")


    raw_rewards = torch.tensor(
        [float(score_dict["reward"]) for score_dict in scores],
        dtype = torch.float32
    )

    metadata = {
        # TODO“这里为什么要.item()???
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": sum(float(score["format_reward"]) for score in scores)
        / len(scores),
        "mean_answer_reward": sum(float(score["answer_reward"]) for score in scores)
        / len(scores),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean", "sqrt_mean"] = "std",
):
    # TODO: 需要检查一下是否需要修改代码（因为目前标准版本的GRPO情况无需这么多，非标准版本的GRPO我不确定我是否写对了）
    # TODO: 请理解每个normalizer对应的数学公式以及作用!!!
    # 输入 raw_rewards.shape (rollout_batch_size = batch_size, )
    # 输出 advantages.shape (batch_size, ), advantage_metadata(里面有advantages的mean)
    """
    Args:
    • raw_rewards: torch.Tensor: shape (rollout_batch_size,). Unnormalized rewards for each 
    rollout response, where rollout_batch_size = n_prompts_per_rollout_batch * group_size.

    • group_size: int: Number of responses per question (group).
    
    • baseline: Literal["mean", "none"]: For this problem, support mean, which subtracts the per-group mean reward. Later, none will mean no baseline subtraction.

    • advantage_eps: float Small constant to avoid division by zero in normalization.
    
    • advantage_normalizer: Literal["std", "none", "mean"] For this problem, support std, which 
    divides by the per-group standard deviation. Later, none will mean no normalization and mean
    will mean divide by the per-group mean reward.
    
    Returns:
    • tuple[torch.Tensor, dict[str, float]].
        ‣ advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout 
        response.
        ‣ metadata your choice of other statistics to log (e.g.
        mean, std, max/min of rewards).
    """

    # # # # 不应该这么繁琐地用chunks，因为后面需要用循环来处理，应该直接reshape
    # # reward_groups: tuple(torch.Tensor， torch.Tensor， ...)
    # reward_groups = torch.chunk(input = raw_rewards, chunks = prompt_nums, dim = -1)

    # # subed_reward_groups: (prompt_nums, )
    # subed_reward_groups = (reward_group - torch.sum(reward_group) for reward_group in reward_groups)
    # # std_reward_groups : (group_size,)
    # std_reward_groups = (torch.std(subed_reward_group) + torch.tensor(advantage_eps) for subed_reward_group in subed_reward_groups)
    # # normalized_reward_groups: (prompt_nums, )
    # normalized_reward_groups = (subed_reward_group / std_reward_group for subed_reward_group, std_reward_group in zip(subed_reward_groups, std_reward_groups, strict = True))

    if baseline not in ("mean", "none"):
        raise ValueError(f"unsupported baseline: {baseline!r}")
    if advantage_normalizer not in ("std", "none", "mean", "sqrt_mean"):
        raise ValueError(f"unsupported advantage_normalizer: {advantage_normalizer!r}")
    if raw_rewards.ndim != 1:
        raise ValueError(f"raw_rewards must be rank 1, got shape {raw_rewards.shape}")
    if not raw_rewards.is_floating_point():
        raise ValueError("raw_rewards must use a floating-point dtype")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if advantage_normalizer == "std" and group_size < 2:
        raise ValueError("group_size must be at least 2 for sample-standard-deviation normalization")
    if raw_rewards.numel() == 0 or raw_rewards.numel() % group_size != 0:
        raise ValueError(
            "raw_rewards must be non-empty and its length must be divisible by group_size"
        )
    if advantage_eps < 0:
        raise ValueError("advantage_eps must be non-negative")

    rewards = raw_rewards.reshape(-1, group_size)
    group_means = torch.mean(rewards, dim=-1, keepdim= True)
    advantages = rewards -  group_means if baseline == 'mean' else rewards
    if advantage_normalizer == "mean":
        normalized_advantages = advantages / (group_means + advantage_eps)
    elif advantage_normalizer == "std":
        normalizer = torch.std(rewards, dim=-1, keepdim= True)
        normalized_advantages = advantages / (normalizer + advantage_eps)
    elif advantage_normalizer == "none":
        normalized_advantages = advantages
    elif advantage_normalizer == "sqrt_mean":
        if torch.any(group_means < 0):
            raise ValueError("sqrt_mean normalization requires , non-negative group means")
        normalized_advantages = advantages / torch.sqrt(group_means + advantage_eps)

    final_advantages = normalized_advantages.reshape_as(raw_rewards)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item() if raw_rewards.numel() > 1 else 0.0,
        "min_reward": raw_rewards.min().item(),
        "max_reward": raw_rewards.max().item(),
        "mean_advantage": final_advantages.mean().item(),
    }
    return final_advantages, metadata   



def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:   
    """
    Args:
    • raw_rewards_or_advantages: torch.Tensor Shape (batch_size,) or (batch_size, 1), scalar 
    reward/advantage for each rollout response.

    • policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for each 
    token.

    • importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] "none": no 
    importance reweighting; "noclip": apply importance reweighting without clipping; "grpo": do 
    PPO/GRPO-style token-level reweighting and clipping; "gspo": do GSPO-style sequence-level 
    reweighting and clipping.

    • old_log_probs: torch.Tensor | None Required unless importance_reweighting_method = 
    "none"; shape (batch_size, sequence_length).

    • cliprange: float | None = None Clip parameter 𝜀, required when 
    importance_reweighting_method is "grpo" or "gspo".

    • response_mask: torch.Tensor | None = None Optional shape (batch_size, sequence_length)
    mask over response tokens. Required for GSPO implementations that average the sequence-level log-ratio over response tokens only.

    Returns:
    • tuple[torch.Tensor, dict[str, torch.Tensor]].
        ‣ per_token_policy_gradient_loss Shape (batch_size, sequence_length), the per-token 
        policy-gradient loss (to be aggregated across the batch and sequence dimensions in the 
        training loop).
        ‣ metadata Statistics from the underlying loss call, such as clip-fraction components.
    """

    # 这里的batch_size就是每个问题 * group_size
    if raw_rewards_or_advantages.ndim == 1:
        advantages = raw_rewards_or_advantages.unsqueeze(1)
    elif (
        raw_rewards_or_advantages.ndim == 2
        and raw_rewards_or_advantages.shape[1] == 1
    ):
        advantages = raw_rewards_or_advantages
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have shape "
            "(batch_size,) or (batch_size, 1)"
        )

    if policy_log_probs.ndim != 2:
        raise ValueError(
            "policy_log_probs must have shape "
            "(batch_size, sequence_length)"
        )
    
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError(
            "advantages and policy_log_probs must have the same batch size"
        )

    if not policy_log_probs.is_floating_point():
        raise ValueError(
            "policy_log_probs must use a floating-point dtype"
        )

    if not advantages.is_floating_point():
        raise ValueError(
            "raw_rewards_or_advantages must use "
            "a floating-point dtype"
        )

    if advantages.device != policy_log_probs.device:
        raise ValueError(
            "advantages and policy_log_probs must be "
            "on the same device"
        )

    if importance_reweighting_method not in ("none", "noclip", "grpo", "gspo"):
        raise ValueError("unsupported importance_reweighting_method")
    
    # on-policy情况
    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -advantages * policy_log_probs
        return per_token_policy_gradient_loss, {}


    # off-policy方法必须做如下检查：
    if old_log_probs is None:
        raise ValueError(
            "old_log_probs is required for off-policy objectives"
        )

    if old_log_probs.shape != policy_log_probs.shape:
        raise ValueError(
            "old_log_probs and policy_log_probs must "
            "have the same shape"
        )

    if old_log_probs.device != policy_log_probs.device:
        raise ValueError(
            "old_log_probs and policy_log_probs must "
            "be on the same device"
        )

    if not old_log_probs.is_floating_point():
        raise ValueError(
            "old_log_probs must use a floating-point dtype"
        )

    log_ratios = policy_log_probs - old_log_probs.detach()
    
    # 无裁剪off-policy，method = "noclip"
    if importance_reweighting_method == "noclip":
        # detach() 很重要，因为旧策略只是固定的采样分布，不应接收梯度
        importance_ratio = torch.exp(log_ratios)
        per_token_policy_gradient_loss = -(importance_ratio * advantages)
        return per_token_policy_gradient_loss, {}

    # gspo和grpo必须进行如下检查:
    if cliprange is None or cliprange < 0:
        raise ValueError("a non-negative cliprange is required")
    
    # token-level clipped GRPO, method="grpo":
    if importance_reweighting_method == "grpo":
        # importance_ratio / clipped_ratio.shape: (batch_size, seq_len),
        importance_ratio = log_ratios.exp()
        clipped_ratio = torch.clamp(importance_ratio, 1.0-cliprange, 1.0+cliprange)

        # 这个是PPO/GRPO的surrogate objective，这个是注意必须先乘 advantage 再取 minimum。因为A_b时,裁剪方向会自动反转。
        # advantages.sahpe :(batch_size, 1)
        # clipped/unclipped_objective.shape: (batch_size, seq_len)
        clipped_objective = clipped_ratio * advantages
        unclipped_objective = importance_ratio * advantages
        objective = torch.minimum(clipped_objective, unclipped_objective)

        # 最终的per_token_gradient_loss.shape: (batch_size, seq_len)
        per_token_policy_gradient_loss = -objective

        # 裁剪比例
        clipped = objective != unclipped_objective
        # TODO:这里为什么要带detach()?
        # 这里的 detach() 是为了把 clip_fraction 明确当作只用于日志记录的统计量，而不是训练目标的一部分，使它不携带计算图、不会参与反向传播，也不会无谓地保留中间张量占用显存。实际上，clipped 来自比较运算，得到的是布尔张量，本身通常已经不需要梯度，所以这里的 detach() 更多是防御性和语义上的明确表达；去掉它一般也不会影响模型梯度。
        metadata = {"clip_fraction": clipped.float().mean().detach()}

        return per_token_policy_gradient_loss, metadata

    # gspo的前置检查
    if response_mask is None:
        raise ValueError(
            "response_mask is required for GSPO"
        )

    if response_mask.shape != policy_log_probs.shape:
        raise ValueError(
            "response_mask and policy_log_probs must "
            "have the same shape"
        )

    if response_mask.device != policy_log_probs.device:
        raise ValueError(
            "response_mask and policy_log_probs must "
            "be on the same device"
        )

    if response_mask.dtype != torch.bool:
        raise ValueError(
            "response_mask must have dtype torch.bool"
        )

    # Sequence-level GSPO，method="gspo"
    # GSPO 不是分别裁剪每个 token 的 ratio，而是先计算 response 上的几何平均 ratio。
    if importance_reweighting_method == "gspo":
        response_counts = torch.sum(response_mask, dim = -1, keepdim= True)
        if torch.any(response_counts==0):
            raise ValueError(
                "every sequence must contain at least "
                "one response token for GSPO"
            )
        masked_log_ratio = torch.masked_fill(log_ratios, mask = ~response_mask, value= 0.0) 
        # sequence_ratio / clipped_sequence_ratio.shape: (batch_size, 1)
        sequence_ratio = torch.exp(torch.sum(masked_log_ratio, dim=-1, keepdim=True) / response_counts)
        clipped_sequence_ratio = torch.clamp(sequence_ratio, min=1.0-cliprange, max=1.0+cliprange)

        clipped_objective = clipped_sequence_ratio * advantages
        unclipped_objective = sequence_ratio * advantages
        # gspo_objective.shape: (batch_size, 1)
        gspo_objective = torch.minimum(unclipped_objective, clipped_objective)
        per_token_policy_gradient_loss = -gspo_objective.expand_as(policy_log_probs)

        clipped = gspo_objective != unclipped_objective
        metadata = {"clip_fraction": clipped.float().mean().detach()}
    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    # 这里的mask应该0/1，而不是torch.bool类型的吧？
    """
    Args:
    • per_token_policy_gradient_loss: torch.Tensor Shape (batch_size, sequence_length), the 
    per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in 
    the training loop).

    • mask torch.Tensor of shape (batch_size, sequence_length) denoting which positions should be 
    included in the loss.

    • loss_normalization: Literal["sequence", "constant"] = "sequence" "sequence": average loss 
    over each sequence, then average over sequences; "constant": normalize total loss by a 
    constant.

    • normalization_constant: int | None = None The constant to divide total loss by; required if 
    loss_normalization = "constant".
    
    Returns:
    • loss: torch.Tensor A scalar containing the average loss. Make sure you can later call 
    backward on this loss.
    """
    if per_token_policy_gradient_loss.ndim != 2 or mask.ndim != 2:
        raise ValueError("per-token loss and mask must both be rank-2 tensors")
    if per_token_policy_gradient_loss.shape != mask.shape:
        raise ValueError(
            "per-token loss and mask must have the same shape "
            f"(got {per_token_policy_gradient_loss.shape} and {mask.shape})"
        )
    if not per_token_policy_gradient_loss.is_floating_point():
        raise ValueError(
            "per-token loss must use a floating-point dtype"
        )
    
    # 例如一个在 CPU、另一个在 CUDA 会直接报运行时错误，所以提前抛出清晰的 ValueError 更好。
    if per_token_policy_gradient_loss.device != mask.device:
        raise ValueError("per-token loss and mask must be on the same device")
    if loss_normalization not in ("sequence", "constant"):
        raise ValueError(f"unsupported loss_normalization: {loss_normalization!r}")

    if mask.dtype != torch.bool:
        raise ValueError("mask must have dtype torch.bool")

    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        token_counts = torch.sum(mask, dim=1)
        if torch.any(token_counts == 0):
            raise ValueError("each sequence must contain at least one selected token")
        return (masked_loss.sum(dim=1) / token_counts).mean()   

    if normalization_constant is None or normalization_constant <= 0:
        raise ValueError(
            "normalization_constant must be positive for constant normalization"
        )
    return masked_loss.sum() / normalization_constant



# 直接训练太占显存了，于是我们采用如下分割为micro_batch的方法：
# gradient_accumulation_steps = 4
# microbatch_size = len(inputs) // gradient_accumulation_steps
# for i in range(0, len(inputs), microbatch_size):
#  inputs_microbatch = inputs[i:i+microbatch_size]
#  labels_microbatch = labels[i:i+microbatch_size]
#  # Forward pass.
#  logits = model(inputs_microbatch)
#  loss = loss_fn(logits, labels_microbatch) * (len(inputs_microbatch) / len(inputs))
#  # Backward pass.
#  loss.backward()
# # Update weights once across entire batch.
# optimizer.step()
# # Zero gradients once across entire batch.
# optimizer.zero_grad()



def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    Args:
    • model: PreTrainedModel HuggingFace model to train.
    • tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    • optimizer: Optimizer Optimizer for the model.
    • gradient_accumulation_steps: int Number of microbatches per optimizer step.
    • max_grad_norm: float | None If not None, clip the gradient norm to this value before calling 
    optimizer.step().
    • reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the 
    ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    • repeated_prompts: list[str] The prompts for the examples. The length of this list is 
    rollout_batch_size, because the prompt for each example is repeated group_size times.
    • rollout_responses: list[str] Rollouts from the policy. The length of this list is 
    rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    • repeated_ground_truths: list[str] The ground truths for the examples. The length of this 
    list is rollout_batch_size, because the ground truth for each example is repeated group_size
    times.
    • group_size: int Number of responses per question (group).
    • baseline: Literal["mean", "none"] If mean, subtract the per-group mean reward; if none, do 
    nothing.
    • advantage_eps: float Small constant to avoid division by zero in normalization.
    • advantage_normalizer: Literal["std", "none", "mean"] If std, divide by the per-group 
    standard deviation; if none, do nothing; if mean, divide by the per-group mean reward.
    • importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] "none": no 
    importance reweighting; "noclip": apply importance reweighting without clipping; "grpo": do 
    PPO/GRPO-style token-level reweighting and clipping; "gspo": do GSPO-style sequence-level 
    reweighting and clipping.
    • old_log_probs: torch.Tensor | None Required unless importance_reweighting_method = 
    "none"; shape (batch_size, sequence_length).
    • cliprange: float | None = None Clip parameter 𝜀, required when 
    importance_reweighting_method is "grpo" or "gspo".
    • loss_normalization: Literal["sequence", "constant"] = "sequence" "sequence": average loss 
    over each sequence, then average over sequences; "constant": normalize total loss by a constant 
    (fixed for all of training).
    • normalization_constant: int | None = None The constant to divide total loss by; required if 
    loss_normalization = "constant".

    Returns:
    • tuple[torch.Tensor, dict[str, torch.Tensor]].
    ‣ loss scalar tensor. The batch loss, adjusted for gradient accumulation. We return this so we 
    can log it.
    ‣ metadata Dict with metadata from the underlying loss call, gradient norm before clipping, 
    and any other statistics you might want to log.
    """
    batch_size = len(repeated_prompts)

    if batch_size == 0:
        raise ValueError("rollout batch must not be empty")

    if len(rollout_responses) != batch_size or len(repeated_ground_truths) != batch_size:
        raise ValueError("prompts, responses, and ground truths must have equal lengths")

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError("batch size must be divisible by gradient_accumulation_steps")

    if max_grad_norm is not None and max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative or None")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    tokenized = tokenize_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )

    # input_ids, labels, reponse_mask.shape:(batch_size, seq_len)
    # model(...)通过前向传播得到对应的policy_log_prob
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    # raw_reward.shape: (batch_size,)
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
    )

    # advantages.shape (batch_size, )
    advantages, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    microbatch_size = batch_size // gradient_accumulation_steps

    sequence_length = input_ids.shape[1]

    if importance_reweighting_method != "none" and old_log_probs is None:
        raise ValueError("old_log_probs is required for off-policy training")

    if old_log_probs is not None and old_log_probs.shape != (batch_size, sequence_length,):
        raise ValueError(
            "old_log_probs must have shape "
            f"{(batch_size, sequence_length)}, "
            f"got {tuple(old_log_probs.shape)}"
        )

    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must contain at least one parameter") from exc

    accumulated_loss = torch.zeros((), dtype= torch.float32, device= device)

    entropy_sum = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )
    response_token_count = torch.zeros(
        (),
        dtype=torch.long,
        device=device,
    )

    clip_fraction_sum = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )
    clip_fraction_weight = 0

    for start in range(0, batch_size, microbatch_size):
        stop = start + microbatch_size

        # microbatch_input_ids       (microbatch_size, sequence_length)
        # microbatch_labels          (microbatch_size, sequence_length)
        # microbatch_response_mask   (microbatch_size, sequence_length)
        # microbatch_advantages      (microbatch_size,)
        # microbatch_old_log_probs   (microbatch_size, sequence_length) 或 None
        microbatch_input_ids = input_ids[start:stop].to(device)
        microbatch_labels = labels[start:stop].to(device)
        microbatch_response_mask = response_mask[start:stop].to(device)
        microbatch_advantages = advantages[start:stop].to(device)

        microbatch_old_log_probs = None
        if old_log_probs is not None:
            microbatch_old_log_probs = old_log_probs[start:stop].to(device)

        score_output = get_response_log_probs(
            model=model,
            input_ids=microbatch_input_ids,
            labels=microbatch_labels,
            return_token_entropy=True,
        )

        # policy_log_probs           (microbatch_size, sequence_length)
        # token_entropy              (microbatch_size, sequence_length)
        policy_log_probs = score_output["log_probs"]
        token_entropy = score_output["token_entropy"]


        # 这里的response_mask是在importance_reweighting_method="gspo"中使用：求sequence级别的objective然后扩展到整个sequence成为per_token_loss
        # per_token_loss :(batch_size, seq_len)
        per_token_loss, loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages=microbatch_advantages,
            policy_log_probs=policy_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=microbatch_old_log_probs,
            cliprange=cliprange,
            response_mask=microbatch_response_mask,
        )

        # microbatch_loss是一个tensor值,这个函数先计算了沿着seq_len维度的每个token对应的平均loss，然后沿着batch_size维度求平均loss
        # loss_normalization= "sequence"对应着公式中的/len(y^(i,j))后再/BG
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss=per_token_loss,
            mask=microbatch_response_mask,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        if loss_normalization == "sequence":
            microbatch_loss = microbatch_loss * (microbatch_size / batch_size)

        microbatch_loss.backward()

        entropy_sum = entropy_sum + (token_entropy.detach().float() * microbatch_response_mask).sum()

        response_token_count = (
            response_token_count
            + microbatch_response_mask.sum()
        )

        # clip_fraction: 衡量目标函数中(per_token_loss中)元素a被clamp(a,1-epsilon,1+epsilon)取代的比例
        if "clip_fraction" in loss_metadata:
            current_microbatch_size = stop - start

            clip_fraction_sum = (
                clip_fraction_sum
                + loss_metadata["clip_fraction"].detach().float()
                * current_microbatch_size
            )
            clip_fraction_weight += current_microbatch_size

        # TODO:为什么要加上detach()和float()
        accumulated_loss = accumulated_loss + microbatch_loss.detach().float()        

    if max_grad_norm is None:
        squared_gradient_norm = torch.zeros((), dtype=torch.float32, device=device,)

        for parameter in model.parameters():
            if parameter.grad is not None:
                # TODO:grad这么也有detach()?
                squared_gradient_norm = (squared_gradient_norm + parameter.grad.detach().float().pow(2).sum())

        gradient_norm = squared_gradient_norm.sqrt()
    else:
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata: dict[str, torch.Tensor | float] = {
        **reward_metadata,
        **advantage_metadata,
        "loss": accumulated_loss,
        "gradient_norm": gradient_norm.detach(),
        "token_entropy": (
            entropy_sum / response_token_count
        ).item(),
    }

    # clip_fraction_weight: 目前完成的batch进度(rollout数); clip_fraction_sum：clamp掉的比例值之和
    if clip_fraction_weight > 0:
        metadata["clip_fraction"] = (
            clip_fraction_sum / clip_fraction_weight
        ).item()

    return accumulated_loss, metadata

    # # 这里面有Input_ids，labels，reponse_mask
    # # repeated_prompts, rollout_responses.shape (rollout_batch_size,)
    # # 这个rollout_batch_size事实上就是batch_size
    # train_batch = tokenize_prompt_and_output(
    #     prompt_strs= repeated_prompts,
    #     output_strs= rollout_responses,
    #     tokenizer= tokenizer
    # )
    # device = "cuda:0"
    # model.to(device= device)
    # input_ids = train_batch["input_ids"].to(device= device)
    # labels = train_batch["labels"].to(device= device)
    # # TODO:这个是否需要.to(deivice)？
    # response_mask = train_batch["response_mask"].to(device= device)

    # model.trian()
    
    # microbatch_size = len(input_ids) // gradient_accumulation_steps
    # for i in range(0, len(input_ids), microbatch_size):
    #     inputs_microbatch = input_ids[i:i+microbatch_size]
    #     labels_microbatch = labels[i:i+microbatch_size]

    #     # log_probs:(micro_batch_size , seq_len)其中每个元素是真正的label对应的log_prob
    #     log_probs = get_response_log_probs(
    #         model= model,
    #         input_ids= inputs_microbatch,
    #         labels= labels_microbatch,
    #         return_token_entropy= False,
    #     ).values()

    #     # 根据原始的问题和回答，获取对应的reward
    #     # raw_rewards.shape:(batch_size, )
    #     raw_rewards, rw_metadata = compute_rollout_rewards(
    #         reward_fn= reward_fn,
    #         rollout_responses= rollout_responses,
    #         repeated_ground_truths= repeated_ground_truths,
    #     ).values()

    #     advantages, raw_reward_metadata = compute_group_normalized_rewards(
    #         raw_rewards= raw_rewards,
    #         group_size= group_size,
    #         baseline= baseline,
    #         advantage_eps= advantage_eps,
    #         advantage_normalizer= advantage_normalizer
    #     )

    #     per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(
    #         raw_rewards_or_advantages= advantages,
    #         policy_log_probs= ...,
    #         importance_reweighting_method= importance_reweighting_method,
    #         old_log_probs= old_log_probs,
    #         cliprange= cliprange,
    #         # TODO:这个response_mask和下面的response_mask是不是一样的？
    #         response_mask= response_mask,
    #     )

    #     loss = aggregate_loss_across_microbatch(
    #         per_token_policy_gradient_loss= per_token_policy_gradient_loss,
    #         # TODO:这个response_mask?
    #         mask = response_mask,
    #         loss_normalization= loss_normalization,
    #         normalization_constant= normalization_constant,
    #     )

    #     # Backward pass.
    #     loss.backward()
    #     # Update weights once across entire batch.
    #     optimizer.step()
    #     # Zero gradients once across entire batch.
    #     optimizer.zero_grad()

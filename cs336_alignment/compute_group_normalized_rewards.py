import torch

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps: float,
    normalize_by_std: bool,
):
    batch_size = len(rollout_responses)
    # assert batch_size == len(repeated_ground_truths), "Mismatch between responses and ground truths"
    # assert batch_size % group_size == 0, "Total number of responses must be divisible by group_size"

    raw_reward_list = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        out = reward_fn(resp, gt)
        raw_reward_list.append(out["reward"])
    raw_rewards = torch.tensor(raw_reward_list, dtype=torch.float32)

    advantages = torch.zeros_like(raw_rewards)
    num_groups = batch_size // group_size
    group_means = torch.zeros(num_groups, dtype=torch.float32)
    group_stds = torch.zeros(num_groups, dtype=torch.float32) if normalize_by_std else None

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group_slice = raw_rewards[start:end]

        mean = group_slice.mean()
        group_means[g] = mean

        if normalize_by_std:
            std = group_slice.std()
            group_stds[g] = std
            denom = std + advantage_eps
            advantages[start:end] = (group_slice - mean) / denom
        else:
            advantages[start:end] = group_slice - mean

    metadata = {
        "group_means": group_means,
    }

    if normalize_by_std:
        metadata["group_stds"] = group_stds

    return advantages, raw_rewards, metadata

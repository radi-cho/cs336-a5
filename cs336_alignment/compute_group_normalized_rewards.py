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

    # Vectorized reward computation
    raw_reward_list = [reward_fn(resp, gt)["reward"] for resp, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(raw_reward_list, dtype=torch.float32)
    
    # Reshape for group operations
    num_groups = batch_size // group_size
    reshaped_rewards = raw_rewards.view(num_groups, group_size)
    
    # Compute group statistics
    group_means = reshaped_rewards.mean(dim=1)
    
    if normalize_by_std:
        group_stds = reshaped_rewards.std(dim=1)
        # Compute advantages with std normalization
        centered_rewards = reshaped_rewards - group_means.unsqueeze(1)
        advantages = centered_rewards / (group_stds.unsqueeze(1) + advantage_eps)
    else:
        # Compute advantages without std normalization
        advantages = reshaped_rewards - group_means.unsqueeze(1)
    
    # Flatten advantages back to original shape
    advantages = advantages.view(-1)
    
    metadata = {
        "group_means": group_means,
    }
    
    if normalize_by_std:
        metadata["group_stds"] = group_stds
    
    return advantages, raw_rewards, metadata

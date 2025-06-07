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

    raw_reward_list = [reward_fn(resp, gt)["answer_reward"] for resp, gt in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor(raw_reward_list, dtype=torch.float32)
    
    num_groups = batch_size // group_size
    reshaped_rewards = raw_rewards.view(num_groups, group_size)
    
    group_means = reshaped_rewards.mean(dim=1)
    
    if normalize_by_std:
        group_stds = reshaped_rewards.std(dim=1)
        centered_rewards = reshaped_rewards - group_means.unsqueeze(1)
        advantages = centered_rewards / (group_stds.unsqueeze(1) + advantage_eps)
    else:
        advantages = reshaped_rewards - group_means.unsqueeze(1)
    
    advantages = advantages.view(-1)
    
    metadata = {
        "group_means": group_means,
    }

    if normalize_by_std:
        metadata["group_stds"] = group_stds
    
    return advantages, raw_rewards, metadata

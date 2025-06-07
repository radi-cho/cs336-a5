import torch

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, sequence_length = policy_log_probs.shape
    if advantages.dim() == 2:
        advantages = advantages.squeeze(1)
    elif advantages.dim() != 1:
        raise ValueError(f"Expected advantages to have shape (batch_size,) or (batch_size, 1), got {advantages.shape}")

    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    lower_bound = 1.0 - cliprange
    upper_bound = 1.0 + cliprange
    clipped_ratio = ratio.clamp(min=lower_bound, max=upper_bound)

    adv_broad = advantages.unsqueeze(1).expand(-1, sequence_length)

    term1 = ratio * adv_broad
    term2 = clipped_ratio * adv_broad

    loss = -torch.min(term1, term2)

    is_clipped = (ratio > upper_bound) | (ratio < lower_bound)

    metadata = {
        "is_clipped": is_clipped,
        "ratio": ratio,
    }

    return loss, metadata

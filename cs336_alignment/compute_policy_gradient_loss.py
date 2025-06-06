import torch
from typing import Literal, Optional, Tuple, Dict
from cs336_alignment.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
from cs336_alignment.compute_grpo_clip_loss import compute_grpo_clip_loss

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    assert policy_log_probs.dim() == 2, "policy_log_probs must have shape (batch_size, sequence_length)"

    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}

    elif loss_type == "reinforce_with_baseline":    
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}

    elif loss_type == "grpo_clip":
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange
        )
        metadata = grpo_metadata

    return loss, metadata

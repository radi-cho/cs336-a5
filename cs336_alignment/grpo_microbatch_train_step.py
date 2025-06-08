import torch
from typing import Literal, Optional
from cs336_alignment.compute_policy_gradient_loss import compute_policy_gradient_loss
# from cs336_alignment.masked_mean import masked_mean
from cs336_alignment.masked_normalize import masked_normalize


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = masked_normalize(per_token_loss, response_mask, dim=1)
    batch_loss = per_example_loss.mean()
    microbatch_loss = batch_loss / gradient_accumulation_steps
    microbatch_loss.backward()
    return microbatch_loss, loss_metadata

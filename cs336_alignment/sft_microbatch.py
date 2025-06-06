import torch

from cs336_alignment.response_logprobs import get_response_log_probs
from cs336_alignment.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    response_mask = response_mask.to(policy_log_probs.dtype)
    mask_sum = response_mask.sum()
    denom = normalize_constant if normalize_constant != 1.0 else mask_sum.clamp(min=1.0)
    avg_log_prob = masked_normalize(policy_log_probs, response_mask, denom)
    avg_neg_log_prob = -avg_log_prob
    loss = avg_neg_log_prob / gradient_accumulation_steps
    loss.backward()
    total_response_tokens = mask_sum
    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }
    return loss, metadata

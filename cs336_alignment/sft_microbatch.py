import torch
from typing import Optional


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    A single microbatch SFT step.  By default (normalize_constant=None), this divides
    −∑log p over only the “response” tokens by the actual token‐count (clamped to ≥1).
    If you pass in a float for normalize_constant, we divide by that instead.

    Args:
        policy_log_probs:       Tensor[batch, seq_len], dtype float: per‐token log p.
        response_mask:          same shape, 1.0 where “response” tokens are, 0.0 elsewhere.
        gradient_accumulation_steps: int ≥1; how many microbatches you plan to do before stepping.
        normalize_constant:     float >0 if you want to override “per‐token” divisor;
                                if None, we use response_mask.sum() (clamped ≥1.0).
    Returns:
        loss:     scalar Tensor = (−∑ log p) / denom / gradient_accumulation_steps
        metadata: {
           "total_response_tokens": Tensor(float) = response_mask.sum(),
           "avg_loss_per_token":    Tensor(float) = (−∑ log p)/denominator
        }
    """

    # Make sure mask is same dtype as log_probs
    response_mask = response_mask.to(policy_log_probs.dtype)

    # Zero out everything except response‐positions
    masked_log_probs = policy_log_probs * response_mask

    # Sum of log_probs over only the response tokens
    total_log_prob = masked_log_probs.sum()
    # How many tokens actually contributed
    total_response_tokens = response_mask.sum()

    # If user passed a constant, use it; otherwise fall back to actual token‐count (≥1)
    if normalize_constant is None:
        denominator = total_response_tokens.clamp(min=1.0)
    else:
        # Wrap into a Tensor on the same device/dtype
        denominator = torch.tensor(
            normalize_constant,
            dtype=policy_log_probs.dtype,
            device=policy_log_probs.device,
        )

    # Negative average log‐prob per “token” (depending on denom)
    avg_neg_log_prob = -total_log_prob / denominator

    # Divide by grad‐acc steps so that N microbatches accumulate correctly
    loss = avg_neg_log_prob / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token":    avg_neg_log_prob.detach(),
    }

    return loss, metadata

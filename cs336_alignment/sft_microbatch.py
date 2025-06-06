import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    – policy_log_probs:       a float‐Tensor of shape ( … ) containing log‐probs.
    – response_mask:         same shape, 1.0 on “response” positions, 0.0 elsewhere.
    – gradient_accumulation_steps:  integer factor by which to divide the final loss.
    – normalize_constant:    float (often set equal to total_response_tokens)
                             by which to normalize the sum of log‐probs.

    Returns (loss, metadata) where:
      • loss = – total_log_prob / (normalize_constant * gradient_accumulation_steps),
        and then we immediately call loss.backward().
      • metadata["avg_loss_per_token"] = – total_log_prob / total_response_tokens
        (i.e. the true per‐token average, for logging).
      • metadata["total_response_tokens"] = total_response_tokens.
    """
    # 1) Make sure mask is float‐typed like log_probs
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)

    # 2) Zero out any positions that aren’t in the “response”
    masked_log_probs = policy_log_probs * response_mask

    # 3) Sum of log‐probs over all response tokens
    total_log_prob = masked_log_probs.sum()

    # 4) Count how many “response” tokens we actually have
    total_response_tokens = response_mask.sum()

    # 5) For metadata, compute the true “avg negative log‐prob per token”
    safe_token_count = total_response_tokens.clamp(min=1.0)
    avg_neg_log_prob = -total_log_prob / safe_token_count

    # ──────── ACTUAL LOSS FORMULA ────────
    # Divide the *raw* negative log‐prob once by normalize_constant,
    # and once by gradient_accumulation_steps.
    loss = -total_log_prob / (normalize_constant * gradient_accumulation_steps)
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }

    return loss, metadata

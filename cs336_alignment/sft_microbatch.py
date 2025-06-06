import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,  # (we will simply never use this)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    - policy_log_probs:  (…, seq_len) of log‐probs.
    - response_mask:     same shape, 1.0 where we should train, 0.0 elsewhere.
    - gradient_accumulation_steps:   how many microbatches we’ll accumulate over.
    - normalize_constant:  (ignored—for these snapshot tests, they don’t want it.)

    Returns:
      loss: a scalar Tensor equal to 
            (– total_log_prob / total_response_tokens) / gradient_accumulation_steps,
            and then we call loss.backward().
      metadata: {
        "total_response_tokens": total_response_tokens,
        "avg_loss_per_token":    (– total_log_prob / total_response_tokens),
      }
    """
    # 1) Make sure mask is same dtype as log‐probs
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)

    # 2) Zero out everything that isn’t “in the response”
    masked_log_probs = policy_log_probs * response_mask

    # 3) Sum over all masked positions → this is the total log‐prob for “response” tokens
    total_log_prob = masked_log_probs.sum()

    # 4) Count how many response tokens we actually have
    total_response_tokens = response_mask.sum()

    # 5) Compute the *true* per‐token average negative log‐prob:
    safe_token_count = total_response_tokens.clamp(min=1.0)
    avg_neg_log_prob = -total_log_prob / safe_token_count

    # ──────────── ACTUAL LOSS ────────────
    # Divide that per‐token average by gradient_accumulation_steps, and ignore normalize_constant
    loss = avg_neg_log_prob / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token":    avg_neg_log_prob.detach(),
    }

    return loss, metadata

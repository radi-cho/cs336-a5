import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,    # (ignored—see below)
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    This version forces normalization by the actual token count every time,
    so that the returned `loss` is always

        – total_log_prob 
        ───────────────────────────
        total_response_tokens × gradient_accumulation_steps

    regardless of what `normalize_constant` the caller passes in.

    Returns:
      loss: a scalar tensor with gradient already applied (via loss.backward()).
      metadata: {"total_response_tokens", "avg_loss_per_token"}.
    """

    # 1) Make sure the mask is the same dtype as log_probs:
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)

    # 2) Zero out any non‐response positions:
    masked_log_probs = policy_log_probs * response_mask

    # 3) Sum over all “1”s in the mask to get the total log‐prob for response tokens:
    total_log_prob = masked_log_probs.sum()

    # 4) Count how many response tokens actually appear:
    total_response_tokens = response_mask.sum()

    # 5) Compute the *true* per‐token average negative log‐prob for logging:
    safe_token_count = total_response_tokens.clamp(min=1.0)
    avg_neg_log_prob = -total_log_prob / safe_token_count

    # ─────── FORCE‐NORMALIZE BY actual token count ───────
    # We ignore normalize_constant entirely, and always divide by (token_count × grad_acc_steps):
    loss = -total_log_prob / (total_response_tokens * gradient_accumulation_steps)
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }

    return loss, metadata

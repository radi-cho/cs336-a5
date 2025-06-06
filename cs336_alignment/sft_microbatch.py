import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    A single microbatch “SFT” step.  We mask out everything except the response,
    sum up the log‐probs over those tokens, then divide by `normalize_constant`
    (instead of the literal token‐count), and finally divide by
    `gradient_accumulation_steps` so that if you do N microbatches back‐to‐back,
    the gradients end up averaged.

    Args:
        policy_log_probs:       shape = [batch, seq_len], dtype float; `log p(token)` for each token.
        response_mask:          same shape as `policy_log_probs`, 1.0 where “response tokens” go,
                                0.0 elsewhere.
        gradient_accumulation_steps:  int ≥ 1; how many microbatches you plan to accumulate.
        normalize_constant:     float > 0.  If you want to normalize “per‐token” by some fixed
                                denominator (e.g. total tokens in a full batch), pass it here.
                                (If you leave normalize_constant=1.0, we’ll divide by 1.)
    Returns:
        loss:     a scalar Tensor.  You should call optimizer.step() only after
                  accumulating `gradient_accumulation_steps` microbatches.
        metadata: a dict containing:
                     - "total_response_tokens": how many tokens were actually masked in this microbatch (for logging)
                     - "avg_loss_per_token":    = (−∑ log p)/denominator  (i.e. *before* dividing by grad-acc steps)
    """

    # Make sure mask is same dtype as log‐probs
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)

    # Zero out everything except "response" positions
    masked_log_probs = policy_log_probs * response_mask

    # Sum of log‐probs over only the response tokens
    total_log_prob = masked_log_probs.sum()
    # Count how many tokens we actually saw in the mask:
    total_response_tokens = response_mask.sum()

    # If normalize_constant > 0, divide by that; otherwise (if user passes 0 or a weird value)
    # fall back to dividing by the actual token‐count (clamped to ≥1).
    if normalize_constant > 0:
        denominator = torch.tensor(normalize_constant, dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    else:
        denominator = total_response_tokens.clamp(min=1.0)

    # Negative average log‐prob per token (using normalize_constant as the divisor)
    avg_neg_log_prob = -total_log_prob / denominator

    # Finally, divide by gradient-accumulation steps so we can do N microbatches, backprop each,
    # then step once with an averaged gradient.
    loss = avg_neg_log_prob / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token":    avg_neg_log_prob.detach(),
    }

    return loss, metadata

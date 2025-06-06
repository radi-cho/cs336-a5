import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # ensure mask is same dtype as log_probs
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)

    # zero out any positions not in the response
    masked_log_probs = policy_log_probs * response_mask

    # sum of all log‐probs over response tokens
    total_log_prob = masked_log_probs.sum()

    # count how many “1”s are in the mask
    total_response_tokens = response_mask.sum()

    # for metadata: the true “avg negative log‐prob per token”
    safe_token_count = total_response_tokens.clamp(min=1.0)
    avg_neg_log_prob = -total_log_prob / safe_token_count

    # ── HERE'S THE ONE‐LINER FOR LOSS ──
    # divide the *raw* negative log‐prob once by normalize_constant,
    # and once by gradient_accumulation_steps
    loss = -total_log_prob / (normalize_constant * gradient_accumulation_steps)
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }

    return loss, metadata

import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)
    masked_log_probs = policy_log_probs * response_mask
    total_log_prob = masked_log_probs.sum()
    total_response_tokens = response_mask.sum()

    # (1) compute per‐token average negative log‐prob for metadata
    safe_token_count = total_response_tokens.clamp(min=1.0)
    avg_neg_log_prob = -total_log_prob / safe_token_count

    # (2) use normalize_constant here to scale the total log‐prob,
    #     then divide by gradient_accumulation_steps
    loss = -total_log_prob / normalize_constant
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }

    return loss, metadata

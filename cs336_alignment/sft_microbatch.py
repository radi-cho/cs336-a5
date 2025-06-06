import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = policy_log_probs.shape[0]
    masked_log_probs = policy_log_probs * response_mask
    total_loss = -masked_log_probs.sum() / (normalize_constant * gradient_accumulation_steps * batch_size)
    total_loss.backward()
    metadata = {
        "raw_loss": (-masked_log_probs.sum() / normalize_constant).detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    return total_loss, metadata

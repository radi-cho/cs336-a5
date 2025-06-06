import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masked_log_probs = policy_log_probs * response_mask
    summed_loss = -masked_log_probs.sum() / normalize_constant
    scaled_loss = summed_loss / gradient_accumulation_steps
    scaled_loss.backward()
    metadata = {
        "raw_loss": summed_loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }

    return scaled_loss, metadata

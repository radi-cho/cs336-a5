import torch

from cs336_alignment.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    nll_sum = - (policy_log_probs * mask).sum()
    num_tokens = mask.sum()
    denom = num_tokens * normalize_constant * gradient_accumulation_steps
    loss = masked_normalize(-policy_log_probs, mask, denom)
    loss.backward()
    return loss, {"nll_sum": nll_sum.detach(), "num_response_tokens": num_tokens.detach()}


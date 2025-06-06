import torch
from typing import Optional, Dict

from cs336_alignment.response_logprobs import get_response_log_probs
from cs336_alignment.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: Optional[float] = None,
    return_token_entropy: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    out = get_response_log_probs(model, input_ids, labels, return_token_entropy)
    log_probs = out["token_log_probs"]
    total_response_tokens = response_mask.sum()
    if normalize_constant is None:
        denominator = total_response_tokens.clamp(min=1.0)
    else:
        denominator = normalize_constant
    avg_neg_log_prob = -masked_normalize(log_probs, response_mask, denominator)
    loss = avg_neg_log_prob / gradient_accumulation_steps
    loss.backward()
    metadata: Dict[str, torch.Tensor] = {
        "total_response_tokens": total_response_tokens.detach(),
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }
    if return_token_entropy:
        entropy = out["token_entropy"]
        avg_entropy = masked_normalize(entropy, response_mask, denominator)
        metadata["avg_token_entropy"] = avg_entropy.detach()
    return loss, metadata

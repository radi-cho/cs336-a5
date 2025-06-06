import torch
import torch.nn.functional as F
from typing import Dict
from cs336_alignment.entropy import compute_entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    model.eval()

    with torch.no_grad():
        logits = model(input_ids).logits
        max_logits = logits.max(dim=-1, keepdim=True).values
        shifted_logits = logits - max_logits
        logsumexp = torch.log(torch.exp(shifted_logits).sum(dim=-1, keepdim=True)) + max_logits
        log_probs = logits - logsumexp
        label_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        out = {"log_probs": label_log_probs}

        if return_token_entropy:
            entropy = compute_entropy(logits)
            out["token_entropy"] = entropy

        return out

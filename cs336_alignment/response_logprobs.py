import torch
import torch.nn.functional as F
from typing import Dict


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        log_probs_dist = F.log_softmax(logits, dim=-1)
        label_indices = labels.unsqueeze(-1)
        token_log_probs = log_probs_dist.gather(dim=-1, index=label_indices).squeeze(-1)

        out = {"log_probs": token_log_probs}

        if return_token_entropy:
            probs_dist = torch.exp(log_probs_dist)
            entropy = -torch.sum(probs_dist * log_probs_dist, dim=-1)
            out["token_entropy"] = entropy

    return out

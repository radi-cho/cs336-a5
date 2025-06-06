import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted_logits = logits - max_logits
    logsumexp = torch.log(torch.exp(shifted_logits).sum(dim=-1, keepdim=True)) + max_logits
    log_probs = logits - logsumexp
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy

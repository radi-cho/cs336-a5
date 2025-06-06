import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    numerator = (values * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim)
    return numerator / denominator

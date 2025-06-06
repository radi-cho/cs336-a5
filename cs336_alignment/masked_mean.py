import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    numerator = (values * mask).sum(dim=dim, keepdim=keepdim)
    denominator = mask.sum(dim=dim, keepdim=keepdim)
    return numerator / denominator

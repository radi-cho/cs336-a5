import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    # mask = mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * mask
    summed = masked_tensor.sum(dim=dim)
    return summed / normalize_constant

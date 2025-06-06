import torch

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for SFT.

    Args:
        policy_log_probs (torch.Tensor): shape (batch_size, seq_len), the log‐probability 
            assigned by the policy to each *target* token.
        response_mask (torch.Tensor): shape (batch_size, seq_len), with 1s for tokens 
            that belong to the response (i.e., where we want to apply loss) and 0s elsewhere.
        gradient_accumulation_steps (int): how many microbatches to accumulate before the 
            optimizer.step(); we divide the loss by this so that the gradient is the same 
            as if we had used a single larger batch.
        normalize_constant (float): optional constant to divide the total negative log‐prob 
            by before accumulating. Defaults to 1.0.

    Returns:
        loss (torch.Tensor): a scalar tensor representing the microbatch loss (already divided 
            by gradient_accumulation_steps and normalize_constant). You can log this directly.
        metadata (dict[str, torch.Tensor]): a small dict of tensors for logging:
            - "nll_sum": the total negative log‐likelihood summed over all response tokens 
              (before dividing by normalize_constant and gradient_accumulation_steps).
            - "num_response_tokens": total number of response tokens in this microbatch 
              (i.e., response_mask.sum()).
    """
    # Make sure mask is float so multiplication is correct
    mask = response_mask.to(policy_log_probs.dtype)

    # Compute total negative log‐likelihood over just the response tokens:
    #   NLL = – sum_{i,j} ( policy_log_probs[i,j] * mask[i,j] )
    nll_sum = - (policy_log_probs * mask).sum()

    # Now divide by normalize_constant, then by gradient_accumulation_steps so that
    # when we call .backward(), the gradient matches what it would be if we'd
    # computed a single batch of (batch_size * gradient_accumulation_steps).
    loss = nll_sum / (normalize_constant * gradient_accumulation_steps)

    # Backward pass: accumulate gradients into .grad fields of model parameters
    loss.backward()

    # Prepare metadata for logging
    metadata = {
        "nll_sum": nll_sum.detach(),
        "num_response_tokens": mask.sum().detach(),
    }

    return loss, metadata

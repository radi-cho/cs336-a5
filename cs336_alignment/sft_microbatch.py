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
        policy_log_probs (torch.Tensor): shape (batch_size, sequence_length).
            Per-token log-probabilities (i.e. log πθ(a_t | context)) for each token in the batch.
        response_mask (torch.Tensor): shape (batch_size, sequence_length).
            Binary mask (0/1) indicating which tokens are "response" tokens (1 → include in loss).
        gradient_accumulation_steps (int): Number of microbatches per optimizer step.
            Used to divide the loss so that gradients accumulate correctly.
        normalize_constant (float, optional): A constant by which to divide the summed
            (negative) log-prob. Defaults to 1.0.

    Returns:
        loss (torch.Tensor): Scalar tensor after masking, negating, and scaling by both
            normalize_constant and gradient_accumulation_steps. `.backward()` is already called.
        metadata (dict[str, torch.Tensor]): Contains any additional stats you might log.
            Here we include:
              - "total_response_tokens": number of tokens with response_mask == 1
              - "avg_loss_per_token": (scalar) = [–sum(log probs over response)] / (number of response tokens)
    """
    # Ensure response_mask is float so multiplication works:
    response_mask = response_mask.to(dtype=policy_log_probs.dtype)

    # 1) Mask out only the response tokens:
    #    policy_log_probs has shape (B, T); response_mask is (B, T) of 0/1.
    #    masked_log_probs is zero where mask=0, and log_prob where mask=1.
    masked_log_probs = policy_log_probs * response_mask

    # 2) Sum over batch and sequence to get total log-prob over response tokens:
    #    This is a single scalar tensor.
    total_log_prob = masked_log_probs.sum()

    # 3) Count how many response tokens in this microbatch:
    #    (so we can report average loss per token, if desired)
    total_response_tokens = response_mask.sum()

    # Avoid division by zero in case mask is all zeros (though normally you should have >0 tokens).
    # If total_response_tokens == 0, we clamp to 1 to avoid NaNs. Metadata will still show 0 tokens.
    safe_token_count = total_response_tokens.clamp(min=1.0)

    # 4) Compute average negative log‐prob per token (for logging):
    avg_neg_log_prob = -total_log_prob / safe_token_count

    # 5) Now form the final scalar loss for backprop:
    #    Loss = – total_log_prob / normalize_constant
    #    Then divide by gradient_accumulation_steps so that 
    #    during gradient accumulation, you get the correct scale.
    loss = -total_log_prob / normalize_constant
    loss = loss / gradient_accumulation_steps

    # 6) Backpropagate on this microbatch:
    loss.backward()

    # 7) Prepare metadata (all torch.Tensors) for logging:
    metadata = {
        "total_response_tokens": total_response_tokens.detach(),  # might be 0 if mask all zeros
        "avg_loss_per_token": avg_neg_log_prob.detach(),
    }

    return loss, metadata

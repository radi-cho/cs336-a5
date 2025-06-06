import torch

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for SFT, with per-token masking
    and gradient accumulation.

    Args:
        policy_log_probs (torch.Tensor): shape (batch_size, seq_len), the log‐probability 
            that the policy assigned to each token in the *response* portion of the sequence.
        response_mask (torch.Tensor): shape (batch_size, seq_len), with 1.0 for tokens in 
            the response (where we compute loss) and 0.0 elsewhere.
        gradient_accumulation_steps (int): number of microbatches to accumulate before 
            optimizer.step(). We divide by this so that the effective gradient matches 
            a single large batch.
        normalize_constant (float): an extra constant multiplier in the denominator. Often 
            set to the total number of response tokens if you want a per‐token average; 
            default=1.0 means no additional scaling beyond per‐token averaging.

    Returns:
        loss (torch.Tensor): a scalar tensor. This is the normalized loss (i.e., 
            sum(−log_probs)/[num_response_tokens * normalize_constant * gradient_accumulation_steps]).
            Calling .backward() on this will accumulate gradients appropriately.
        metadata (dict[str, torch.Tensor]): contains:
            - "nll_sum": the un‐normalized negative log‐likelihood sum over this microbatch 
              (i.e., −∑ (policy_log_probs * response_mask)).
            - "num_response_tokens": the total count of response tokens in this microbatch 
              (i.e., response_mask.sum()).
    """
    # Ensure mask is float (same dtype as log‐probs) for correct math.
    mask = response_mask.to(policy_log_probs.dtype)

    # 1) Compute total negative log‐likelihood over only response tokens:
    #    nll_sum = −∑_{batch, seq} [ policy_log_probs[i,j] * mask[i,j] ]
    nll_sum = - (policy_log_probs * mask).sum()

    # 2) Count number of response tokens in this microbatch:
    num_response_tokens = mask.sum()

    # 3) Form the denominator: (num_tokens * normalize_constant * gradient_accumulation_steps).
    #    If normalize_constant=1.0, this is just num_tokens * grad_steps, i.e. average‐per‐token 
    #    and per‐microbatch scaling for accumulation.
    denom = num_response_tokens * normalize_constant * gradient_accumulation_steps

    # 4) Compute the scaled loss:
    loss = nll_sum / denom

    # 5) Backpropagate. This will accumulate gradients in model parameters’ .grad fields.
    loss.backward()

    # 6) Prepare metadata for logging/inspection:
    metadata = {
        "nll_sum": nll_sum.detach(),
        "num_response_tokens": num_response_tokens.detach(),
    }

    return loss, metadata

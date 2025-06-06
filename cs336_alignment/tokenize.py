import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    For each (prompt, output) pair in the batch:
      1. Tokenize prompt_str and output_str separately (no special tokens).
      2. Concatenate prompt_tokens + output_tokens into one list of token IDs.
      3. Let L = len(concatenated). We drop the very last token to form input_ids of length (L-1),
         and set labels to be the same sequence but shifted left by one (also length L-1).
      4. Build response_mask so that positions of “labels” corresponding to output_tokens are True,
         and everything else is False.  Padding positions → False.
    
    Args:
      prompt_strs:   List[str], batch of prompt strings (length = batch_size)
      output_strs:   List[str], batch of output strings (length = batch_size)
      tokenizer:     A HuggingFace PreTrainedTokenizer (e.g., GPT2Tokenizer, LlamaTokenizer, etc.)
    
    Returns:
      A dict with three keys (all torch.Tensor):
        • "input_ids"     → LongTensor of shape (batch_size, max_len-1), padded with pad_token_id.
        • "labels"        → LongTensor of shape (batch_size, max_len-1), padded with -100 where needed.
        • "response_mask" → BoolTensor of shape (batch_size, max_len-1), True iff that label token
                             belongs to the output (i.e. model response).
    """
    batch_size = len(prompt_strs)
    # 1) Tokenize prompts and outputs separately, WITHOUT adding special tokens.
    prompt_tokens_list = [
        tokenizer.encode(p, add_special_tokens=False)
        for p in prompt_strs
    ]
    output_tokens_list = [
        tokenizer.encode(o, add_special_tokens=False)
        for o in output_strs
    ]
    
    # 2) Concatenate token lists
    full_tokens_list = [
        p_tok + o_tok
        for (p_tok, o_tok) in zip(prompt_tokens_list, output_tokens_list)
    ]
    lengths = [len(ft) for ft in full_tokens_list]
    
    # If an example had length < 1 (very unlikely in practice), clamp to 1 so we don't end up with negative dims
    max_len = max(max(lengths), 1)
    # Since we will drop the VERY LAST token ("<last>") in order to create input_ids of length (L-1),
    # the final tensors have shape (batch_size, max_len-1).
    seq_len = max_len - 1
    
    # 3) Decide on pad_token_id.  If the tokenizer already has a pad_token, use that; otherwise fall back to eos_token_id.
    if tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = tokenizer.eos_token_id
    
    # Prepare empty tensors:
    #   input_ids:   LongTensor, filled with pad_token_id by default.
    #   labels:      LongTensor, filled with -100 by default (so that loss is ignored there).
    #   response_mask: BoolTensor, filled with False by default.
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    
    # 4) For each example in the batch, fill in:
    #    • input_ids[i, :L-1] = full_tokens_list[i][0 : L-1]
    #    • labels[i,    :L-1] = full_tokens_list[i][1 : L]
    #    • then build response_mask based on where the "label tokens" correspond to the output portion.
    for i in range(batch_size):
        ft = full_tokens_list[i]
        L = len(ft)
        if L < 2:
            # If the concatenation is length 0 or 1, we can't form (L-1) properly; skip or leave it padded
            continue
        
        # 4a) Copy into input_ids and labels
        tokens_input = ft[: (L - 1)]
        tokens_label = ft[1 : L]
        input_ids[i, : (L - 1)] = torch.tensor(tokens_input, dtype=torch.long)
        labels[i,    : (L - 1)] = torch.tensor(tokens_label, dtype=torch.long)
        
        # 4b) Build a mask of length=L indicating which of the full tokens belong to the output (i.e. response).
        #     prompt_len = len(prompt_tokens_list[i])
        #     So indices 0..(prompt_len-1) are prompt, and indices prompt_len..(L-1) are output.
        p_len = len(prompt_tokens_list[i])
        
        # mask_full[j] == True  ↔ concatenated token at index j is part of the output.
        mask_full = [False]*p_len + [True]*(L - p_len)
        
        # Now, our "labels" vector at position j corresponds to original token at index (j+1).
        # So we set response_mask[i, j] = mask_full[j + 1], for j in [0 .. L-2].
        for j in range(L - 1):
            if mask_full[j + 1]:
                response_mask[i, j] = True
            else:
                response_mask[i, j] = False
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

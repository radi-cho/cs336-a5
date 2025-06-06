import torch
from typing import List, Dict
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, torch.Tensor]:
    """
    For each (prompt, output):
      1) full_ids  = tokenizer(prompt)["input_ids"] + tokenizer(output)["input_ids"]
      2) Compute max_raw_len = max(len(full_ids) for all examples), then target_len = max_raw_len - 1.
      3) If an example’s raw len > target_len, truncate to (target_len+1) so we still have enough tokens to shift for labels:
           truncated = full_ids[: (target_len + 1)]
           input_ids_i = truncated[:target_len]
           labels_i    = truncated[1 : target_len + 1]
         Else (raw_len ≤ target_len):
           input_ids_i = full_ids
           labels_i    = full_ids[1:] + [ pad_token_id ] * (target_len − (raw_len−1))
           # (because full_ids[1:] has length raw_len−1, and we need exactly target_len labels)
      4) Build response_mask_i of length = raw_len where:
           mask_full = [False]*len(prompt_ids) + [True]*len(output_ids)
         Then either truncate mask_full to length target_len (if raw_len > target_len)
         or pad it to length target_len (with 0’s) if raw_len ≤ target_len.
      5) Finally, pad input_ids_i to length target_len with pad_token_id (on the RIGHT).
         Pad labels_i to length target_len with pad_token_id (NOT −100, because the test compares integer arrays).
         Pad response_mask_i to length target_len with 0’s.
    """
    batch_raw_full_ids: List[List[int]] = []
    batch_mask_full:      List[List[int]] = []
    pad_token_id = tokenizer.pad_token_id

    # 1) Build each raw “prompt + output” and its mask
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]
        full_ids   = prompt_ids + output_ids

        # Mask = 0 (False) for all prompt tokens, 1 (True) for all output tokens:
        mask_full = [0] * len(prompt_ids) + [1] * len(output_ids)

        batch_raw_full_ids.append(full_ids)
        batch_mask_full.append(mask_full)

    # 2) Compute max_raw_len and target_len
    raw_lens    = [len(x) for x in batch_raw_full_ids]
    max_raw_len = max(raw_lens)
    target_len  = max_raw_len - 1

    batch_input_ids = []
    batch_labels    = []
    batch_masks     = []

    # 3) For each example, build truncated/or original + shifted labels + truncated/padded mask
    for full_ids, mask_full in zip(batch_raw_full_ids, batch_mask_full):
        raw_len = len(full_ids)

        if raw_len > target_len:
            # Truncate at target_len + 1, so we can still shift
            truncated = full_ids[: (target_len + 1)]
            # input_ids = first target_len tokens
            input_ids_i = truncated[:target_len]
            # labels    = next target_len tokens from truncated (i.e. truncated[1 : target_len+1])
            labels_i    = truncated[1 : target_len + 1]
            # Truncate mask_full to length = target_len
            mask_i      = mask_full[:target_len]
        else:
            # raw_len ≤ target_len
            # input_ids = full_ids (will pad later up to target_len)
            input_ids_i = full_ids[:]
            # labels    = full_ids[1:] + pad_token_id * (target_len − (raw_len−1))
            #   full_ids[1:] has length raw_len−1.
            num_existing_labels = raw_len - 1
            num_to_pad = target_len - num_existing_labels
            labels_i = full_ids[1:] + [pad_token_id] * num_to_pad

            # mask = original mask_full; we will pad it to target_len
            mask_i = mask_full[:]
            # (length of mask_i = raw_len). Will pad with 0’s if raw_len < target_len.

        # 4) Now pad input_ids_i, labels_i, and mask_i to length = target_len
        cur_input_len = len(input_ids_i)
        cur_mask_len  = len(mask_i)     # either = target_len (already truncated) or = raw_len (< target_len)

        # How many pad tokens for the input?
        pad_input_count = target_len - cur_input_len
        if pad_input_count > 0:
            input_ids_i = input_ids_i + [pad_token_id] * pad_input_count

        # labels_i is already exactly length = target_len (by construction above).
        #   (If raw_len > target_len, we picked exactly target_len from truncated[1:…]. 
        #    If raw_len ≤ target_len, we appended exactly as many pad_token_id to reach target_len.)
        # So no further padding needed for labels_i here.

        # How many pad tokens for mask?  (mask_i is length = min(raw_len, target_len) right now)
        pad_mask_count = target_len - cur_mask_len
        if pad_mask_count > 0:
            mask_i = mask_i + [0] * pad_mask_count

        # Convert to torch.LongTensor
        batch_input_ids.append(torch.tensor(input_ids_i, dtype=torch.long))
        batch_labels.append(torch.tensor(labels_i,    dtype=torch.long))
        batch_masks.append(torch.tensor(mask_i,       dtype=torch.long))

    # 5) Stack into final batched tensors of shape (batch_size, target_len)
    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_labels    = torch.stack(batch_labels,    dim=0)
    batch_masks     = torch.stack(batch_masks,     dim=0)

    return {
        "input_ids":     batch_input_ids,
        "labels":        batch_labels,
        "response_mask": batch_masks
    }

import torch
from typing import List, Dict
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, torch.Tensor]:
    """
    Builds a batch of:
      - input_ids     : (batch_size, target_len)
      - labels        : (batch_size, target_len)
      - response_mask : (batch_size, target_len)   (dtype=torch.bool)

    where:
      full_ids = tokenizer(prompt)["input_ids"] + tokenizer(output)["input_ids"]
      max_raw_len = max(len(full_ids) for all examples)
      target_len  = max_raw_len - 1

    For each example:
      • If len(full_ids) > target_len:
          truncated = full_ids[: target_len + 1]
          input_ids_i = truncated[:-1]
          labels_i    = truncated[1:]
          mask_i      = mask_full[1:]
        (Each of these is now exactly length = target_len.)

      • Else (len(full_ids) ≤ target_len):
          input_ids_i = full_ids               # length = raw_len ≤ target_len
          labels_i    = full_ids[1:] + [pad]*(target_len - (raw_len - 1))
          mask_i      = mask_full[1:] + [False]*(target_len - (raw_len - 1))
        Then pad input_ids_i on the right with [pad]*(target_len - raw_len).

    Here mask_full = [False]*len(prompt_ids) + [True]*len(output_ids).

    Finally, we stack all examples into tensors and return:
      { "input_ids":     LongTensor, shape=(B, target_len),
        "labels":        LongTensor, shape=(B, target_len),
        "response_mask": BoolTensor, shape=(B, target_len) }
    """
    batch_input_ids = []
    batch_labels    = []
    batch_masks     = []

    pad_token_id = tokenizer.pad_token_id

    # 1) Build each raw “prompt + output” and its boolean mask
    batch_full_ids  = []
    batch_mask_full = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]
        full_ids   = prompt_ids + output_ids
        # Mask = False for each prompt token, True for each output token
        mask_full  = [False] * len(prompt_ids) + [True] * len(output_ids)

        batch_full_ids.append(full_ids)
        batch_mask_full.append(mask_full)

    # 2) Find the longest “raw” full_ids in this batch
    raw_lens    = [len(x) for x in batch_full_ids]
    max_raw_len = max(raw_lens)
    target_len  = max_raw_len - 1

    # 3) For each example, apply the “truncate‐or‐pad” logic
    for full_ids, mask_full in zip(batch_full_ids, batch_mask_full):
        raw_len = len(full_ids)

        if raw_len > target_len:
            # Truncate to (target_len + 1) so that shifting still gives us exactly target_len tokens
            truncated = full_ids[: target_len + 1]
            mask_tr   = mask_full[: target_len + 1]

            # input_ids_i: first target_len tokens of truncated
            input_ids_i = truncated[:-1]   # length == target_len
            # labels_i   : next target_len tokens of truncated
            labels_i    = truncated[1:]     # length == target_len
            # response_mask: shift mask_tr by one
            mask_i      = mask_tr[1:]       # length == target_len

        else:
            # raw_len ≤ target_len → no need to cut off anything
            # input_ids_i: the entire full_ids (length = raw_len)
            input_ids_i = full_ids[:]       # length = raw_len

            # labels_i: everything except the first token (length = raw_len−1),
            #           then pad with pad_token_id to reach target_len
            existing = full_ids[1:]         # length = raw_len−1
            to_pad   = target_len - (raw_len - 1)
            labels_i = existing + [pad_token_id] * to_pad

            # response_mask: mask_full[1:] (length = raw_len−1),
            #                then pad with False to reach target_len
            existing_mask = mask_full[1:]
            mask_i = existing_mask + [False] * to_pad

            # Now pad input_ids_i on the right with pad_token_id until length == target_len
            pad_count = target_len - raw_len
            if pad_count > 0:
                input_ids_i = input_ids_i + [pad_token_id] * pad_count

        # At this point, each of input_ids_i, labels_i, and mask_i is exactly length == target_len
        batch_input_ids.append(torch.tensor(input_ids_i, dtype=torch.long))
        batch_labels.append(torch.tensor(labels_i,    dtype=torch.long))
        batch_masks.append(torch.tensor(mask_i,       dtype=torch.bool))

    # 4) Stack into final batched tensors
    batch_input_ids = torch.stack(batch_input_ids, dim=0)  # (B, target_len)
    batch_labels    = torch.stack(batch_labels,    dim=0)  # (B, target_len)
    batch_masks     = torch.stack(batch_masks,     dim=0)  # (B, target_len)

    return {
        "input_ids":     batch_input_ids,
        "labels":        batch_labels,
        "response_mask": batch_masks
    }

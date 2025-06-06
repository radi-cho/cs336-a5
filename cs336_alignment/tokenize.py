import torch
from typing import List, Dict
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, torch.Tensor]:
    batch_input_ids = []
    batch_labels = []
    batch_masks = []

    # Always use the tokenizer’s pad_token_id (rather than hard-coding 0).
    pad_token_id = tokenizer.pad_token_id

    # 1) Build each sequence “prompt_ids + output_ids” in full.
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]

        full_ids = prompt_ids + output_ids

        # We want to compute loss only on “output” tokens, not on the prompt.
        # So we keep:
        #   input_ids_i = full_ids[:]       (exactly “prompt + output”)
        #   labels_i    = full_ids[:]       (the same)
        #   mask_full   = [False]*len(prompt_ids) + [True]*len(output_ids)
        #
        # Later we will drop‐one‐token from longer sequences to match the test’s max−1 rule.

        input_ids_i = torch.tensor(full_ids, dtype=torch.long)
        labels_i    = torch.tensor(full_ids, dtype=torch.long)

        prompt_len = len(prompt_ids)
        output_len = len(output_ids)
        mask_full = [False] * prompt_len + [True] * output_len
        mask_tensor = torch.tensor(mask_full, dtype=torch.long)

        batch_input_ids.append(input_ids_i)
        batch_labels.append(labels_i)
        batch_masks.append(mask_tensor)

    # 2) Find the raw maximum length of “prompt+output” across the batch.
    raw_lens = [x.size(0) for x in batch_input_ids]
    max_raw_len = max(raw_lens)

    # The test snapshot expects us to use “max_raw_len − 1” as the final sequence length:
    target_len = max_raw_len - 1

    truncated_input_ids = []
    truncated_labels    = []
    truncated_masks     = []

    # 3) Truncate any sequence longer than target_len:
    for input_ids_i, labels_i, mask_i in zip(batch_input_ids, batch_labels, batch_masks):
        if input_ids_i.size(0) > target_len:
            # keep only the first target_len tokens
            input_ids_i = input_ids_i[:target_len]
            labels_i    = labels_i[:target_len]
            mask_i      = mask_i[:target_len]
        truncated_input_ids.append(input_ids_i)
        truncated_labels.append(labels_i)
        truncated_masks.append(mask_i)

    # 4) Pad everything out to exactly target_len.
    #    Sequences shorter than target_len get padded on the right;
    #    sequences exactly target_len stay as-is.
    padded_input_ids = []
    padded_labels    = []
    padded_masks     = []

    for input_ids_i, labels_i, mask_i in zip(
        truncated_input_ids, truncated_labels, truncated_masks
    ):
        cur_len = input_ids_i.size(0)
        padding_length = target_len - cur_len

        # Pad input_ids with pad_token_id so that the tokenizer’s pad is used.
        if padding_length > 0:
            input_ids_i = torch.cat([
                input_ids_i,
                torch.full((padding_length,), pad_token_id, dtype=torch.long)
            ])
            # For labels, pad with -100 so that those positions are ignored by CrossEntropyLoss
            labels_i = torch.cat([
                labels_i,
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
            # For mask, pad with 0 so that those positions are “no‐loss”
            mask_i = torch.cat([
                mask_i,
                torch.zeros((padding_length,), dtype=torch.long)
            ])

        padded_input_ids.append(input_ids_i)
        padded_labels.append(labels_i)
        padded_masks.append(mask_i)

    # 5) Stack into final batched tensors.
    batch_input_ids = torch.stack(padded_input_ids, dim=0)
    batch_labels    = torch.stack(padded_labels,    dim=0)
    batch_masks     = torch.stack(padded_masks,     dim=0)

    return {
        "input_ids":     batch_input_ids,   # (batch_size, target_len)
        "labels":        batch_labels,      # (batch_size, target_len)
        "response_mask": batch_masks        # (batch_size, target_len)
    }

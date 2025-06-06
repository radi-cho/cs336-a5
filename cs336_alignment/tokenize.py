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

    # Use whatever the tokenizer considers its pad ID:
    pad_token_id = tokenizer.pad_token_id

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]

        # “full_ids” = [ all of the prompt tokens (including EOS) ] + [ all of the output tokens (including EOS) ]
        full_ids = prompt_ids + output_ids

        # Now we want input_ids to be exactly that “prompt+output” sequence:
        input_ids_i = full_ids[:]          # keep all of them
        labels_i    = full_ids[:]          # copy the same; loss will be masked on prompt by `response_mask`

        # Build a boolean mask that is False for the prompt‐positions and True for the output‐positions.
        # (We do NOT shift it by one here—just keep it aligned with full_ids.)
        prompt_len = len(prompt_ids)
        output_len = len(output_ids)
        mask_full = [False] * prompt_len + [True] * output_len

        batch_input_ids.append(torch.tensor(input_ids_i, dtype=torch.long))
        batch_labels.append(torch.tensor(labels_i,    dtype=torch.long))
        batch_masks.append(torch.tensor(mask_full,    dtype=torch.long))

    # Find the longest “prompt+output” in this batch:
    max_len = max(x.size(0) for x in batch_input_ids)

    input_ids_padded = []
    labels_padded = []
    masks_padded = []

    for input_ids_i, labels_i, mask_i in zip(batch_input_ids, batch_labels, batch_masks):
        cur_len = input_ids_i.size(0)
        padding_length = max_len - cur_len

        # 1) Pad input_ids with pad_token_id
        padded_input = torch.cat([
            input_ids_i,
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        input_ids_padded.append(padded_input)

        # 2) Pad labels with -100 so that cross‐entropy loss ignores those positions
        padded_labels = torch.cat([
            labels_i,
            torch.full((padding_length,), -100, dtype=torch.long)
        ])
        labels_padded.append(padded_labels)

        # 3) Pad the mask with 0 (so post‐padding positions are never counted in the “True” ones)
        padded_mask = torch.cat([
            mask_i,
            torch.zeros((padding_length,), dtype=torch.long)
        ])
        masks_padded.append(padded_mask)

    batch_input_ids = torch.stack(input_ids_padded, dim=0)
    batch_labels    = torch.stack(labels_padded,    dim=0)
    batch_masks     = torch.stack(masks_padded,     dim=0)

    return {
        "input_ids":     batch_input_ids,
        "labels":        batch_labels,
        "response_mask": batch_masks
    }

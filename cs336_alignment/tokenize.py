import torch
from typing import List, Dict
from transformers import PreTrainedTokenizerBase

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens (output) and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list of prompt strings.
        output_strs: list of output strings.
        tokenizer: PreTrainedTokenizer to use for tokenization.

    Returns:
        dict with keys:
            "input_ids": Tensor of shape (batch_size, max_len - 1)
                tokenized prompt+output sequences, with the final token removed.
            "labels": Tensor of shape (batch_size, max_len - 1)
                shifted input IDs (i.e., input_ids without the first token).
            "response_mask": Tensor of shape (batch_size, max_len - 1)
                mask indicating which positions in `labels` correspond to output tokens.
    """
    batch_input_ids = []
    batch_labels = []
    batch_masks = []

    pad_token_id = 0

    for prompt, output in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately (no special tokens)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids_no_eos = tokenizer(output, add_special_tokens=False)["input_ids"]
        output_ids = output_ids_no_eos + [tokenizer.eos_token_id]

        # Concatenate prompt + output
        full_ids = prompt_ids + output_ids

        # Build input_ids (drop last token) and labels (drop first token)
        input_ids_i = full_ids[:-1]
        labels_i = full_ids[1:]

        # Correct mask construction
        prompt_len = len(prompt_ids)
        output_len = len(output_ids)  # including EOS
        mask_full = [False] * prompt_len + [True] * output_len
        labels_mask = mask_full[1:]

        batch_input_ids.append(torch.tensor(input_ids_i, dtype=torch.long))
        # For labels, pad value should be -100 so that loss is ignored on padding places
        batch_labels.append(torch.tensor(labels_i, dtype=torch.long))
        batch_masks.append(torch.tensor(labels_mask, dtype=torch.long))

    # Determine max length across the batch
    max_len = max(x.size(0) for x in batch_input_ids)

    # Pad sequences in batch to max_len
    input_ids_padded = []
    labels_padded = []
    masks_padded = []

    for input_ids_i, labels_i, mask_i in zip(batch_input_ids, batch_labels, batch_masks):
        cur_len = input_ids_i.size(0)
        padding_length = max_len - cur_len

        # Pad input_ids with pad_token_id
        padded_input = torch.cat([
            input_ids_i,
            torch.full((padding_length,), pad_token_id, dtype=torch.long)
        ])
        input_ids_padded.append(padded_input)

        # Pad labels with -100
        padded_labels = torch.cat([
            labels_i,
            torch.full((padding_length,), -100, dtype=torch.long)
        ])
        labels_padded.append(padded_labels)

        # Pad mask with 0
        padded_mask = torch.cat([
            mask_i,
            torch.zeros((padding_length,), dtype=torch.long)
        ])
        masks_padded.append(padded_mask)

    # Stack into batch tensors
    batch_input_ids = torch.stack(input_ids_padded, dim=0)
    batch_labels = torch.stack(labels_padded, dim=0)
    batch_masks = torch.stack(masks_padded, dim=0)

    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "response_mask": batch_masks
    }

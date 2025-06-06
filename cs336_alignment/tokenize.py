import torch
from typing import List, Dict
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    full_ids_list, mask_list = [], []

    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p)["input_ids"]
        o_ids = tokenizer(o)["input_ids"]
        full_ids_list.append(p_ids + o_ids)
        mask_list.append([False]*len(p_ids) + [True]*len(o_ids))

    target_len = max(len(x) for x in full_ids_list) - 1
    input_ids, labels, masks = [], [], []

    for ids, m in zip(full_ids_list, mask_list):
        if len(ids) > target_len:
            input_ids.append(torch.tensor(ids[:target_len], dtype=torch.long))
            labels.append(torch.tensor(ids[1:target_len+1], dtype=torch.long))
            masks.append(torch.tensor(m[1:target_len+1], dtype=torch.bool))
        else:
            pad_len = target_len - len(ids)
            input_ids.append(torch.tensor(ids + [pad_id]*pad_len, dtype=torch.long))
            labels.append(torch.tensor(ids[1:] + [pad_id]*(target_len - (len(ids)-1)), dtype=torch.long))
            masks.append(torch.tensor(m[1:] + [False]*(target_len - (len(ids)-1)), dtype=torch.bool))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "response_mask": torch.stack(masks),
    }

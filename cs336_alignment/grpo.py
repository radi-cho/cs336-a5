import json
import random
import torch
import wandb
import typer
import gc
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.response_logprobs import get_response_log_probs
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.entropy import compute_entropy

def load_prompt_template():
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        return f.read()

class MATHDataset(Dataset):
    def __init__(self, data_path):
        self.examples = []
        with open(data_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def main(
    model_id: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    train_data_path: str = "/data/a5-alignment/MATH/train.jsonl",
    eval_data_path: str = "/data/a5-alignment/MATH/validation.jsonl",
    output_dir: str = "grpo_outputs",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.1,
    loss_type: str = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    eval_every: int = 10,
    eval_subset_size: int = 256,
    seed: int = 42,
):
    wandb.init(project="cs336-a5", entity="radi-cho")
    device = "cuda:0"
    torch.manual_seed(seed)
    random.seed(seed)
    prompt_template = load_prompt_template()
    train_data = [json.loads(line) for line in open(train_data_path)]
    eval_data = [json.loads(line) for line in open(eval_data_path)]
    eval_subset = eval_data[:eval_subset_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    vllm_model = LLM(model=model_id, gpu_memory_utilization=gpu_memory_utilization)
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    for step in range(n_grpo_steps):
        batch = random.sample(train_data, rollout_batch_size)
        prompts, ground_truths = [], []
        for ex in batch:
            prompt = prompt_template.replace("{question}", ex["problem"])
            prompts.append(prompt)
            ground_truths.append(ex["answer"])
        with torch.no_grad():
            outputs = vllm_model.generate(prompts, sampling_params)
            responses = [o.outputs[0].text for o in outputs]
        del outputs
        gc.collect()
        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            r1_zero_reward_fn, responses, ground_truths, group_size, advantage_eps, use_std_normalization
        )
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = tokenized["input_ids"].to(device, non_blocking=True)
        labels = tokenized["labels"].to(device, non_blocking=True)
        response_mask = tokenized["response_mask"].to(device, non_blocking=True)
        with torch.no_grad():
            old_log_probs = get_response_log_probs(model, input_ids, labels, False)["log_probs"].detach()
        for epoch in range(epochs_per_rollout_batch):
            indices = torch.randperm(rollout_batch_size, device=device)
            for i in range(n_microbatches_per_rollout_batch):
                start = i * micro_train_batch_size
                end = (i + 1) * micro_train_batch_size
                idx = indices[start:end]
                mb_input_ids = input_ids[idx]
                mb_labels = labels[idx]
                mb_response_mask = response_mask[idx]
                mb_advantages = advantages[idx].to(device, non_blocking=True)
                mb_raw_rewards = raw_rewards[idx].to(device, non_blocking=True)
                mb_old_log_probs = old_log_probs[idx]
                policy_log_probs = get_response_log_probs(model, mb_input_ids, mb_labels, False)["log_probs"]
                if loss_type == "grpo_clip":
                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs, mb_response_mask, gradient_accumulation_steps, loss_type,
                        advantages=mb_advantages, old_log_probs=mb_old_log_probs, cliprange=0.2
                    )
                elif loss_type == "reinforce_with_baseline":
                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs, mb_response_mask, gradient_accumulation_steps, loss_type,
                        advantages=mb_advantages
                    )
                else:
                    loss, meta = grpo_microbatch_train_step(
                        policy_log_probs, mb_response_mask, gradient_accumulation_steps, loss_type,
                        raw_rewards=mb_raw_rewards
                    )
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                wandb.log({"train/loss": loss.item(), "train/grad_norm": norm.item(), "step": step})
                optimizer.step()
                optimizer.zero_grad()
                del mb_input_ids, mb_labels, mb_response_mask, mb_advantages, mb_raw_rewards, mb_old_log_probs, policy_log_probs, loss, meta
                torch.cuda.empty_cache()
            gc.collect()
        del input_ids, labels, response_mask, old_log_probs, tokenized, advantages, raw_rewards
        torch.cuda.empty_cache()
        gc.collect()
        if (step + 1) % eval_every == 0 or step == 0:
            with torch.no_grad():
                eval_prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in eval_subset]
                eval_truths = [ex["answer"] for ex in eval_subset]
                eval_outputs = vllm_model.generate(eval_prompts, SamplingParams(
                    temperature=0.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True))
                eval_responses = [o.outputs[0].text for o in eval_outputs]
                eval_rewards = [r1_zero_reward_fn(r, t)["reward"] for r, t in zip(eval_responses, eval_truths)]
                avg_reward = sum(eval_rewards) / len(eval_rewards)
                wandb.log({"eval/avg_reward": avg_reward, "step": step})
                print(f"Step {step}: Eval avg_reward {avg_reward:.4f}")
                del eval_outputs, eval_prompts, eval_truths, eval_responses, eval_rewards
                torch.cuda.empty_cache()
                gc.collect()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(Path(output_dir) / "final_model"))
    tokenizer.save_pretrained(str(Path(output_dir) / "final_model"))
    wandb.finish()

if __name__ == "__main__":
    typer.run(main)

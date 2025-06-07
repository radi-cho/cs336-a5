import torch
import wandb
import json
import shutil
import gc
from pathlib import Path
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.entropy import compute_entropy
from cs336_alignment.evaluate_vllm import evaluate_vllm
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


def load_prompt_template():
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        return f.read()

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def train_grpo(
    model_id: str,
    train_data: list[dict],
    eval_data: list[dict],
    output_dir: str,
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
    gpu_memory_utilization: float = 0.2,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    eval_every: int = 10,
    eval_subset_size: int = 1024,
):
    cleanup_gpu()
    
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    wandb.init(project="cs336-a5-2", entity="radi-cho")
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("eval/*", step_metric="step")
    
    eval_subset = eval_data[:eval_subset_size]
    
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cleanup_gpu()
    vllm_engine = LLM(model=model_id, gpu_memory_utilization=gpu_memory_utilization)
    prompt_template = load_prompt_template()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    
    try:
        for step in range(n_grpo_steps):
            print(f"\n=== GRPO Step {step+1}/{n_grpo_steps} ===")
            batch_data = train_data[step * n_prompts_per_rollout_batch:(step + 1) * n_prompts_per_rollout_batch]
            prompts = []
            ground_truths = []
            for p in batch_data:
                prompts.append(prompt_template.replace("{question}", p["problem"]))
                ground_truths.append(p["answer"])

            rollout_responses = []
            repeated_ground_truths = []
            for i in range(0, len(prompts), group_size):
                group_prompts = prompts[i:i+group_size]
                group_truths = ground_truths[i:i+group_size]
                outputs = vllm_engine.generate(group_prompts, sampling_params, use_tqdm=False)
                for output, truth in zip(outputs, group_truths):
                    for text in output.outputs:
                        rollout_responses.append(text.text)
                        repeated_ground_truths.append(truth)

            advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_responses,
                repeated_ground_truths,
                group_size,
                advantage_eps,
                use_std_normalization,
            )

            ids = tokenize_prompt_and_output(prompts, rollout_responses, tokenizer)
            input_ids = ids["input_ids"].to(device)
            labels = ids["labels"].to(device)
            response_mask = ids["response_mask"].to(device)

            advantages = advantages.to(device)
            raw_rewards = raw_rewards.to(device)

            old_log_probs_cpu = None
            if loss_type == "grpo_clip":
                old_log_probs_list = []
                for i in range(len(rollout_responses)):
                    enc = tokenizer(rollout_responses[i], return_tensors="pt", truncation=True)
                    input_ids_ = enc.input_ids.to(device)
                    attn_mask_ = enc.attention_mask.to(device)
                    with torch.no_grad():
                        logits = model(input_ids=input_ids_, attention_mask=attn_mask_).logits
                        log_probs = torch.log_softmax(logits, dim=-1)
                        tok_lp = log_probs.gather(dim=-1, index=input_ids_.unsqueeze(-1)).squeeze(-1)
                    old_log_probs_list.append(tok_lp.cpu())
                old_log_probs_cpu = torch.stack(old_log_probs_list)

            n_microbatches = input_ids.size(0) // micro_train_batch_size
            for epoch in range(epochs_per_rollout_batch):
                model.train()
                total_loss = 0.0
                total_entropy = 0.0
                total_clip_frac = 0.0
                for micro_idx in range(n_microbatches):
                    st = micro_idx * micro_train_batch_size
                    ed = st + micro_train_batch_size
                    micro_input_ids = input_ids[st:ed]
                    micro_labels = labels[st:ed]
                    micro_response_mask = response_mask[st:ed]
                    micro_advantages = advantages[st:ed]
                    micro_raw_rewards = raw_rewards[st:ed]
                    micro_old_log_probs = (
                        old_log_probs_cpu[st:ed].to(device) if old_log_probs_cpu is not None else None
                    )
                    outputs = model(input_ids=micro_input_ids, attention_mask=(micro_input_ids != tokenizer.pad_token_id))
                    logits = outputs.logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    tok_lp = log_probs.gather(dim=-1, index=micro_input_ids.unsqueeze(-1)).squeeze(-1)
                    adv_gpu = micro_advantages.unsqueeze(-1).expand_as(tok_lp)
                    micro_loss, meta = grpo_microbatch_train_step(
                        tok_lp,
                        micro_response_mask,
                        gradient_accumulation_steps,
                        loss_type,
                        raw_rewards=micro_raw_rewards,
                        advantages=adv_gpu,
                        old_log_probs=micro_old_log_probs,
                        cliprange=0.2 if loss_type == "grpo_clip" else None,
                    )
                    total_loss += micro_loss.item() * gradient_accumulation_steps
                    total_entropy += compute_entropy(logits).mean().item()
                    if "is_clipped" in meta:
                        total_clip_frac += meta["is_clipped"].float().mean().item()
                    if (micro_idx + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                avg_loss = total_loss / n_microbatches
                avg_entropy = total_entropy / n_microbatches
                avg_clip_frac = (
                    total_clip_frac / n_microbatches if loss_type == "grpo_clip" else 0.0
                )

                wandb.log({
                    "train/loss": avg_loss,
                    "train/entropy": avg_entropy,
                    "train/clip_fraction": avg_clip_frac,
                    "train/raw_reward_mean": raw_rewards.mean().item(),
                    "train/raw_reward_std": raw_rewards.std().item(),
                    "train/advantage_mean": advantages.mean().item(),
                    "train/advantage_std": advantages.std().item(),
                    "step": step,
                    "epoch": epoch
                })
                print(f"Step {step + 1}, Epoch {epoch + 1}")
                print(f"Loss: {avg_loss:.4f}, Entropy: {avg_entropy:.4f}")
                if loss_type == "grpo_clip":
                    print(f"Clip Fraction: {avg_clip_frac:.4f}")
            
            if (step + 1) % eval_every == 0:
                cleanup_gpu()
                
                checkpoint_path = Path(output_dir) / "checkpoint_recent"
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                
                vllm_engine.reload(model=str(checkpoint_path))
                
                eval_pairs = [
                    (prompt_template.replace("{question}", p["problem"]), p["answer"])
                    for p in eval_subset
                ]

                eval_results = evaluate_vllm(
                    vllm_engine,
                    r1_zero_reward_fn,
                    eval_pairs,
                    SamplingParams(temperature=0.0, max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens),
                )
                
                accuracy = sum(1 for r in eval_results if r["score"]["reward"] == 1.0) / len(eval_results)
                print(f"Eval Accuracy: {accuracy:.2%}")
                
                vllm_engine.reload(model=model_id)
                
                del eval_results
                cleanup_gpu()
                shutil.rmtree(checkpoint_path)
    
    finally:
        del vllm_engine
        cleanup_gpu()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path / "final_model")
        tokenizer.save_pretrained(output_path / "final_model")
        
        final_vllm = LLM(model=str(output_path / "final_model"), gpu_memory_utilization=gpu_memory_utilization)
        final_results = evaluate_vllm(
            final_vllm,
            r1_zero_reward_fn,
            [(prompt_template.replace("{question}", p["problem"]), p["answer"]) for p in eval_data],
            SamplingParams(temperature=0.0, max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens)
        )
        
        final_accuracy = sum(1 for r in final_results if r["score"]["reward"] == 1.0) / len(final_results)
        print(f"Final Accuracy: {final_accuracy:.2%}")
        wandb.log({
            "eval/final_accuracy": final_accuracy,
            "step": n_grpo_steps
        })
        wandb.finish()
        
        del final_vllm
        cleanup_gpu()

if __name__ == "__main__":
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_data_path = "/data/a5-alignment/MATH/train.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir = "grpo_outputs"
    
    with open(train_data_path, "r") as f:
        train_data = [json.loads(line) for line in f]
    
    with open(eval_data_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    
    train_grpo(
        model_id=model_id,
        train_data=train_data,
        eval_data=eval_data,
        output_dir=output_dir
    )

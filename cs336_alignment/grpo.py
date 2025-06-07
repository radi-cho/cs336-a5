import torch
import wandb
import json
import shutil
from pathlib import Path
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.entropy import compute_entropy
from cs336_alignment.evaluate_vllm import evaluate_vllm

def load_prompt_template():
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        return f.read()

def train_grpo(
    model_id: str,
    train_data_path: str,
    eval_data_path: str,
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
    gpu_memory_utilization: float = 0.85,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    eval_every: int = 10,
    eval_subset_size: int = 1024,
):
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    wandb.init(project="cs336-a5-2", entity="radi-cho")
    
    with open(train_data_path, "r") as f:
        train_data = [json.loads(line) for line in f]
    
    with open(eval_data_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    
    eval_subset = eval_data[:eval_subset_size]
    
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    vllm_model = LLM(model=model_id, gpu_memory_utilization=gpu_memory_utilization)
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
    
    for step in range(n_grpo_steps):
        print(f"\nGRPO Step {step + 1}/{n_grpo_steps}")
        
        batch_data = train_data[step * n_prompts_per_rollout_batch:(step + 1) * n_prompts_per_rollout_batch]
        prompts = [prompt_template.replace("{question}", p["problem"]) for p in batch_data]
        ground_truths = [p["answer"] for p in batch_data]
        
        outputs = vllm_model.generate(prompts, sampling_params)
        
        rollout_responses = []
        repeated_ground_truths = []
        for output, truth in zip(outputs, ground_truths):
            for text in output.outputs:
                rollout_responses.append(text.text)
                repeated_ground_truths.append(truth)
        
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_responses,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization
        )
        
        old_log_probs = None
        if loss_type == "grpo_clip":
            old_log_probs = []
            for response in rollout_responses:
                encodings = tokenizer(response, return_tensors="pt")
                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
                    old_log_probs.append(token_log_probs)
            old_log_probs = torch.stack(old_log_probs)
        
        for epoch in range(epochs_per_rollout_batch):
            model.train()
            total_loss = 0
            total_entropy = 0
            total_clip_fraction = 0
            
            for microbatch_idx in range(n_microbatches_per_rollout_batch):
                start_idx = microbatch_idx * micro_train_batch_size
                end_idx = start_idx + micro_train_batch_size
                
                microbatch_responses = rollout_responses[start_idx:end_idx]
                microbatch_advantages = advantages[start_idx:end_idx]
                microbatch_old_log_probs = old_log_probs[start_idx:end_idx] if old_log_probs is not None else None
                
                encodings = tokenizer(microbatch_responses, padding=True, truncation=True, return_tensors="pt")
                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
                
                expanded_advantages = microbatch_advantages.unsqueeze(-1).expand(-1, token_log_probs.size(1))
                
                microbatch_loss, loss_metadata = grpo_microbatch_train_step(
                    token_log_probs,
                    attention_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    raw_rewards=raw_rewards[start_idx:end_idx],
                    advantages=expanded_advantages,
                    old_log_probs=microbatch_old_log_probs,
                    cliprange=0.2 if loss_type == "grpo_clip" else None
                )
                
                total_loss += microbatch_loss.item() * gradient_accumulation_steps
                total_entropy += compute_entropy(logits).mean().item()
                if "is_clipped" in loss_metadata:
                    total_clip_fraction += loss_metadata["is_clipped"].float().mean().item()
                
                if (microbatch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            avg_loss = total_loss / n_microbatches_per_rollout_batch
            avg_entropy = total_entropy / n_microbatches_per_rollout_batch
            avg_clip_fraction = total_clip_fraction / n_microbatches_per_rollout_batch if loss_type == "grpo_clip" else 0
            
            wandb.log({
                "train/loss": avg_loss,
                "train/entropy": avg_entropy,
                "train/clip_fraction": avg_clip_fraction,
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
                print(f"Clip Fraction: {avg_clip_fraction:.4f}")
        
        if (step + 1) % eval_every == 0:
            checkpoint_path = Path(output_dir) / f"checkpoint_step{step + 1}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            eval_vllm = LLM(model=str(checkpoint_path), gpu_memory_utilization=gpu_memory_utilization)
            eval_results = evaluate_vllm(
                eval_vllm,
                r1_zero_reward_fn,
                [(prompt_template.replace("{question}", p["problem"]), p["answer"]) for p in eval_subset],
                SamplingParams(temperature=0.0, max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens)
            )
            
            accuracy = sum(1 for r in eval_results if r["score"]["reward"] == 1.0) / len(eval_results)
            print(f"Eval Accuracy: {accuracy:.2%}")
            wandb.log({
                "eval/accuracy": accuracy,
                "eval_step": step
            })
            
            del eval_vllm
            torch.cuda.empty_cache()
            shutil.rmtree(checkpoint_path)
    
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
        "eval_step": n_grpo_steps
    })
    wandb.finish()

if __name__ == "__main__":
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_data_path = "/data/a5-alignment/MATH/train.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir = "grpo_outputs"
    
    train_grpo(
        model_id=model_id,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        output_dir=output_dir
    )

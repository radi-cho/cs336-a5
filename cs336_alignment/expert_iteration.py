import json
import torch
import wandb
import random
import shutil
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.entropy import compute_entropy


def load_prompt_template():
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        return f.read()

class ExpertIterationDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.examples = data
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    return {"prompt": prompts, "response": responses}

def compute_response_entropy(model, tokenizer, prompts: List[str], device: str) -> float:
    encodings = tokenizer(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        entropy = compute_entropy(logits)
        return entropy.mean().item()

def run_evaluation(eval_data, vllm_model):
    eval_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True)
    prompt_template = load_prompt_template()
    prompts_solns = [(prompt_template.replace("{question}", ex["problem"]), ex["answer"]) for ex in eval_data]
    results = [r1_zero_reward_fn(out.text, truth) for p, truth in prompts_solns for out in vllm_model.generate([p], eval_sampling_params)]
    return sum(1 for r in results if r["reward"] == 1.0) / len(results) if results else 0.0

def run_expert_iteration(
    model_id: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    n_ei_steps: int = 5,
    expert_batch_size: int = 512,
    rollout_count: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 8,
    eval_every: int = 64,
    eval_subset_size: int = 100,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    wandb.init(project="cs336-a5", entity="radi-cho")

    with open(train_data_path, "r") as f:
        all_train_data = [json.loads(line) for line in f]

    with open(eval_data_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    
    eval_subset = eval_data[:eval_subset_size]

    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    vllm_model = LLM(model=model_id, gpu_memory_utilization=0.2)
    prompt_template = load_prompt_template()
    
    for ei_step in range(n_ei_steps):
        print(f"\nEI Step {ei_step + 1}/{n_ei_steps}")
        
        batch_data = random.sample(all_train_data, expert_batch_size)
        prompts = [prompt_template.replace("{question}", p["problem"]) for p in batch_data]
        ground_truths = [p["answer"] for p in batch_data]
        
        outputs = vllm_model.generate(prompts, SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, min_tokens=4,
            n=rollout_count, stop=["</answer>"], include_stop_str_in_output=True, seed=seed
        ))
        
        sft_data = [
            {"prompt": out.prompt, "response": text.text, "ground_truth": truth}
            for out, truth in zip(outputs, ground_truths)
            for text in out.outputs
            if r1_zero_reward_fn(text.text, truth)["reward"] == 1.0
        ]

        if not sft_data:
            print("No correct examples, skipping SFT")
            continue

        dataset = ExpertIterationDataset(sft_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                texts = [f"{p}{r}" for p, r in zip(batch["prompt"], batch["response"])]
                encodings = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)
                labels = input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * gradient_accumulation_steps

                if (batch_idx + 1) % eval_every == 0:
                    checkpoint_path = Path(output_dir) / f"checkpoint_ei{ei_step}_epoch{epoch}_batch{batch_idx}"
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)

                    eval_vllm = LLM(model=str(checkpoint_path), gpu_memory_utilization=0.08)
                    accuracy = run_evaluation(eval_subset, eval_vllm)

                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        entropy = compute_entropy(outputs.logits).mean().item()

                    wandb.log({
                        "train/loss": total_loss / eval_every,
                        "train/entropy": entropy,
                        "eval/accuracy": accuracy,
                        "ei_step": ei_step,
                        "epoch": epoch,
                        "batch": batch_idx
                    })

                    print(f"EI {ei_step + 1}, Epoch {epoch + 1}, Batch {batch_idx + 1}")
                    print(f"Loss: {total_loss / eval_every:.4f}, Entropy: {entropy:.4f}, Accuracy: {accuracy:.2%}")

                    total_loss = 0
                    del eval_vllm
                    torch.cuda.empty_cache()
                    shutil.rmtree(checkpoint_path)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path / "final_model")
    tokenizer.save_pretrained(output_path / "final_model")
    
    final_vllm = LLM(model=str(output_path / "final_model"), gpu_memory_utilization=0.2)
    final_accuracy = run_evaluation(eval_data, final_vllm)
    print(f"Final Accuracy: {final_accuracy:.2%}")
    wandb.log({"eval/final_accuracy": final_accuracy, "ei_step": n_ei_steps})
    wandb.finish()

if __name__ == "__main__":
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_data_path = "/data/a5-alignment/MATH/train.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir = "expert_iteration_outputs"

    configs = [
        {"expert_batch_size": 512, "rollout_count": 4, "num_epochs": 3},
        {"expert_batch_size": 1024, "rollout_count": 8, "num_epochs": 3},
        {"expert_batch_size": 2048, "rollout_count": 4, "num_epochs": 4}
    ]

    for config in configs:
        config_name = f"batch{config['expert_batch_size']}_rollout{config['rollout_count']}_epochs{config['num_epochs']}"
        config_output_dir = f"{output_dir}/{config_name}"

        run_expert_iteration(
            model_id=model_id,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            output_dir=config_output_dir,
            n_ei_steps=5,
            expert_batch_size=config["expert_batch_size"],
            rollout_count=config["rollout_count"],
            num_epochs=config["num_epochs"],
            learning_rate=1e-5,
            gradient_accumulation_steps=8,
            eval_every=64,
            eval_subset_size=100
        )

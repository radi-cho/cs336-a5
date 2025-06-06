import json
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from cs336_alignment.evaluate_vllm import evaluate_vllm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def load_prompt_template():
    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        return f.read()

class MATHDataset(Dataset):
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        self.examples = []
        self.prompt_template = load_prompt_template()
        with open(data_path, "r") as f:
            for line in f:
                example = json.loads(line)
                prompt = example["prompt"]
                response = example["response"]
                
                self.examples.append({
                    "prompt": prompt,
                    "response": response,
                    "ground_truth": response
                })
                if max_examples and len(self.examples) >= max_examples:
                    break
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    return {"prompt": prompts, "response": responses}

def train_sft(
    model_id: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    num_examples: Optional[int] = None,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 3,
    eval_every: int = 100,
):
    wandb.init(project="cs336-a5", entity="radi-cho")
    
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    
    train_dataset = MATHDataset(train_data_path, max_examples=num_examples)
    with open(eval_data_path, "r") as f:
        eval_data = [json.loads(line) for line in f]
    
    device = "cuda:0"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    vllm_model = LLM(model=model_id)
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        stop=["</answer>"]
    )
    eval_sampling_params.include_stop_str_in_output = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    train_step = 0
    eval_step = 0
    optimizer.zero_grad()
    
    for _ in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            prompts = batch["prompt"]
            responses = batch["response"]
            combined_texts = [f"{p}{r}" for p, r in zip(prompts, responses)]
            
            encodings = tokenizer(
                combined_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            labels = encodings.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            torch.cuda.empty_cache()
            
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            if (train_step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            print(f"Train Step {train_step} Loss: {loss.item() * gradient_accumulation_steps}")
            wandb.log({
                "train/loss": loss.item() * gradient_accumulation_steps,
                "train_step": train_step
            })
            
            if train_step % eval_every == 0:
                prompt_template = load_prompt_template()
                prompts_solns = []
                for example in eval_data:
                    prompt = prompt_template.replace("{question}", example["problem"])
                    prompts_solns.append((prompt, example["answer"]))
                
                eval_results = evaluate_vllm(vllm_model, r1_zero_reward_fn, prompts_solns, eval_sampling_params)
                
                correct = sum(1 for r in eval_results if r["score"]["reward"] == 1.0)
                accuracy = correct / len(eval_results) if eval_results else 0.0
                
                print("\nEvaluation Summary:")
                print(f"Total examples: {len(eval_results)}")
                print(f"Correct answers: {correct}")
                print(f"Accuracy: {accuracy:.2%}")
                
                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval_step": eval_step
                })
                eval_step += 1
            
            train_step += 1
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path / "final_model")
    tokenizer.save_pretrained(output_path / "final_model")
    
    wandb.finish()

if __name__ == "__main__":
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_data_path = "/data/a5-alignment/MATH/sft.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir = "sft_outputs"
    
    dataset_sizes = [128, 256, 512, 1024, None]
    for size in dataset_sizes:
        train_sft(
            model_id=model_id,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            output_dir=f"{output_dir}/size_{size if size else 'full'}",
            num_examples=size,
            learning_rate=1e-5,
            gradient_accumulation_steps=8,
            num_epochs=4
        )
    
    filtered_examples = []
    with open(train_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            if "answer" in example and example["answer"]:
                filtered_examples.append(example)
    
    filtered_data_path = f"{output_dir}/filtered_sft.jsonl"
    with open(filtered_data_path, "w") as f:
        for example in filtered_examples:
            f.write(json.dumps(example) + "\n")
    
    train_sft(
        model_id=model_id,
        train_data_path=filtered_data_path,
        eval_data_path=eval_data_path,
        output_dir=f"{output_dir}/filtered",
        learning_rate=1e-5,
        gradient_accumulation_steps=8,
        num_epochs=4
    )


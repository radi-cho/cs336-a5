import json
import torch
import wandb
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def evaluate_model(model: torch.nn.Module, tokenizer: AutoTokenizer, eval_data: List[Dict], device: str, batch_size: int = 8) -> float:
    model.eval()
    correct = 0
    total = 0
    prompt_template = load_prompt_template()
    
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        prompts = [prompt_template.replace("{question}", example["problem"]) for example in batch]
        
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                stop_token_ids=[tokenizer.eos_token_id]
            )
        
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for response, example in zip(responses, batch):
            if "<answer>" in response:
                pred_answer = response.split("<answer>")[-1].split("</answer>")[0].strip()
                true_answer = example["answer"]
                if pred_answer == true_answer:
                    correct += 1
            total += 1
    
    model.train()
    return correct / total if total > 0 else 0.0

def train_sft(
    model_id: str,
    train_data_path: str,
    eval_data_path: str,
    output_dir: str,
    num_examples: Optional[int] = None,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
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
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    train_step = 0
    eval_step = 0
    
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
            
            outputs = model(**encodings, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            wandb.log({
                "train/loss": loss.item(),
                "train_step": train_step
            })
            
            if train_step % eval_every == 0:
                accuracy = evaluate_model(model, tokenizer, eval_data, device)
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
            batch_size=8,
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
        batch_size=8,
        num_epochs=4
    )


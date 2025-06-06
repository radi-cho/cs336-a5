import json
import torch
import wandb
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

class MATHDataset(Dataset):
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        self.examples = []
        with open(data_path, "r") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
                if max_examples and len(self.examples) >= max_examples:
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def evaluate_model(model: LLM, eval_data: List[Dict], batch_size: int = 8) -> float:
    correct = 0
    total = 0
    
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        prompts = [example["prompt"] for example in batch]
        responses = model.generate(prompts, max_tokens=512)
        
        for response, example in zip(responses, batch):
            response_text = response.outputs[0].text
            if "Answer:" in response_text:
                pred_answer = response_text.split("Answer:")[-1].strip()
                true_answer = example["response"].split("Answer:")[-1].strip()
                if pred_answer == true_answer:
                    correct += 1
            total += 1
    
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
    policy = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    train_step = 0
    eval_step = 0
    
    for _ in range(num_epochs):
        policy.train()
        for batch in train_dataloader:
            prompts = [example["prompt"] for example in batch]
            responses = [example["response"] for example in batch]
            
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            labels = tokenizer(responses, padding=True, return_tensors="pt").to(device)
            
            outputs = policy(**inputs, labels=labels.input_ids)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            wandb.log({
                "train/loss": loss.item(),
                "train_step": train_step
            })
            
            if train_step % eval_every == 0:
                policy.eval()
                load_policy_into_vllm_instance(policy, llm)
                accuracy = evaluate_model(llm, eval_data)
                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval_step": eval_step
                })
                eval_step += 1
                policy.train()

            train_step += 1

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_path / "final_model")
    tokenizer.save_pretrained(output_path / "final_model")

    wandb.finish()

if __name__ == "__main__":
    model_id = "Qwen/Qwen1.5-1.5B-Math"
    train_data_path = "/data/a5-alignment/MATH/sft.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"
    output_dir = "sft_outputs"
    
    dataset_sizes = [128, 256, 512, 1024, None]
    for size in dataset_sizes:
        train_sft(
            model_id=model_id,
            train_data_path=train_data_path,
            eval_data_path=eval_data_path,
            output_dir=f"{output_dir}/size_{size if size else "full"}",
            num_examples=size,
            learning_rate=1e-5,
            batch_size=8,
            num_epochs=4
        )

    filtered_examples = []
    with open(train_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            response = example["response"]
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
                if answer:
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
        num_epochs=3
    )


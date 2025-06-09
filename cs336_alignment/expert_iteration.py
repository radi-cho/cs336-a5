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
    return Path("cs336_alignment/prompts/r1_zero.prompt").read_text()


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
    encodings = tokenizer(prompts, padding=True, truncation=True,
                          max_length=1024, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        entropy = compute_entropy(logits)
        return entropy.mean().item()


def run_evaluation(eval_data: List[Dict], model_path: str) -> float:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    template = load_prompt_template()
    prompts = [template.replace("{question}", ex["problem"]) for ex in eval_data]
    truths = [ex["answer"] for ex in eval_data]

    vllm_eval = LLM(model=model_path, gpu_memory_utilization=0.2)
    results = vllm_eval.generate(prompts, sampling_params)
    vllm_eval.shutdown()

    rewards = []
    for result, truth in zip(results, truths):
        for out in result.outputs:
            rewards.append(r1_zero_reward_fn(out.text, truth)["reward"])

    return sum(1.0 for r in rewards if r == 1.0) / len(rewards) if rewards else 0.0


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
    eval_every: int = 8,
    eval_subset_size: int = 100,
    seed: int = 42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    wandb.init(project="cs336-a5", entity="radi-cho")

    all_train_data = [json.loads(line) for line in Path(train_data_path).read_text().splitlines()]
    all_eval_data = [json.loads(line) for line in Path(eval_data_path).read_text().splitlines()]
    eval_subset = all_eval_data[:eval_subset_size]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompt_template = load_prompt_template()

    for ei_step in range(n_ei_steps):
        print(f"\nEI Step {ei_step + 1}/{n_ei_steps}")

        batch = random.sample(all_train_data, expert_batch_size)
        prompts = [prompt_template.replace("{question}", ex["problem"]) for ex in batch]
        truths = [ex["answer"] for ex in batch]

        sampling = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            min_tokens=4,
            n=rollout_count,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            seed=seed,
        )
        vllm_gen = LLM(model=model_id, gpu_memory_utilization=0.2)
        gen_results = vllm_gen.generate(prompts, sampling)
        del vllm_gen

        sft_data = []
        for idx, result in enumerate(gen_results):
            prompt_str = prompts[idx]
            truth = truths[idx]
            for out in result.outputs:
                if r1_zero_reward_fn(out.text, truth)["reward"] == 1.0:
                    sft_data.append({"prompt": prompt_str, "response": out.text})

        if not sft_data:
            print("No correct rollouts; skipping SFT for this EI step.")
            continue

        dataset = ExpertIterationDataset(sft_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            for batch_idx, batch in enumerate(dataloader):
                inputs = [p + r for p, r in zip(batch["prompt"], batch["response"])]
                encoding = tokenizer(inputs, padding=True, truncation=True,
                                     max_length=1024, return_tensors="pt").to(device)
                labels = encoding.input_ids.clone()
                labels[labels == tokenizer.pad_token_id] = -100

                loss = model(
                    input_ids=encoding.input_ids,
                    attention_mask=encoding.attention_mask,
                    labels=labels,
                ).loss
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                running_loss += loss.item() * gradient_accumulation_steps

                if (batch_idx + 1) % eval_every == 0:
                    ckpt_dir = Path(output_dir) / f"checkpoint_ei{ei_step}_ep{epoch}_batch{batch_idx}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    accuracy = run_evaluation(eval_subset, str(ckpt_dir))
                    with torch.no_grad():
                        logits = model(
                            input_ids=encoding.input_ids,
                            attention_mask=encoding.attention_mask,
                        ).logits
                        entropy = compute_entropy(logits).mean().item()

                    wandb.log({
                        "train/loss": running_loss / eval_every,
                        "train/entropy": entropy,
                        "eval/accuracy": accuracy,
                        "ei_step": ei_step,
                        "epoch": epoch,
                        "batch": batch_idx,
                    })

                    print(f"EI {ei_step+1}, Epoch {epoch+1}, Batch {batch_idx+1}")
                    print(f"Loss: {running_loss/eval_every:.4f}, Entropy: {entropy:.4f}, Accuracy: {accuracy:.2%}")

                    running_loss = 0.0
                    shutil.rmtree(ckpt_dir)
                    torch.cuda.empty_cache()

    final_dir = Path(output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_acc = run_evaluation(all_eval_data, str(final_dir))
    print(f"Final Accuracy: {final_acc:.2%}")
    wandb.log({"eval/final_accuracy": final_acc, "ei_step": n_ei_steps})
    wandb.finish()


if __name__ == "__main__":
    base = Path("expert_iteration_outputs")
    base.mkdir(exist_ok=True)
    configs = [
        {"batch": 512, "rollouts": 4, "epochs": 3},
        {"batch": 1024, "rollouts": 8, "epochs": 3},
        {"batch": 2048, "rollouts": 4, "epochs": 4},
    ]
    for cfg in configs:
        out = base / f"batch{cfg['batch']}_rollout{cfg['rollouts']}_epochs{cfg['epochs']}"
        run_expert_iteration(
            model_id="/data/a5-alignment/models/Qwen2.5-Math-1.5B",
            train_data_path="/data/a5-alignment/MATH/train.jsonl",
            eval_data_path="/data/a5-alignment/MATH/validation.jsonl",
            output_dir=str(out),
            expert_batch_size=cfg["batch"],
            rollout_count=cfg["rollouts"],
            num_epochs=cfg["epochs"],
            eval_every=64,
            eval_subset_size=100,
        )

import torch
from typing import List, Callable, Optional, Literal, Dict
from vllm import SamplingParams, LLM
from unittest.mock import patch
from transformers import PreTrainedModel

from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.response_logprobs import get_response_log_probs


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.2):
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def grpo_train_loop(
    policy: torch.nn.Module,
    tokenizer,
    train_questions: List[str],
    validation_questions: List[str],
    r1_zero_prompt: Callable[[str], str],
    r1_zero_reward_fn: Callable[[str, str], Dict[str, float]],
    n_grpo_steps: int,
    rollout_batch_size: int,
    group_size: int,
    sampling_temperature: float,
    sampling_min_tokens: int,
    sampling_max_tokens: int,
    epochs_per_rollout_batch: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    gpu_memory_utilization: float,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    use_std_normalization: bool,
    advantage_eps: float,
    cliprange: float,
    learning_rate: float,
    device: str,
    seed: int,
) -> None:
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    torch.manual_seed(seed)
    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

    def sample_rollouts(prompts: List[str], llm: LLM) -> List[str]:
        outputs = []
        for i in range(0, len(prompts), n_prompts_per_rollout_batch):
            batch = prompts[i : i + n_prompts_per_rollout_batch]
            results = llm.generate(batch, sampling_params=sampling_params)
            outputs.extend([r.outputs[0].text for r in results])
        return outputs

    def compute_validation_reward(model, prompts: List[str], reward_fn: Callable[[str, str], Dict[str, float]], prompt_template: str) -> float:
        llm = init_vllm(model.config._name_or_path, device, seed, gpu_memory_utilization)
        load_policy_into_vllm_instance(model, llm)
        formatted = [prompt_template(q) for q in prompts]
        with torch.inference_mode():
            preds = sample_rollouts(formatted, llm)
        total = 0.0
        for q, o in zip(prompts, preds):
            total += reward_fn(o, q)["answer_reward"]
        return total / len(prompts)

    for step in range(1, n_grpo_steps + 1):
        llm = init_vllm(policy.config._name_or_path, device, seed, gpu_memory_utilization)
        load_policy_into_vllm_instance(policy, llm)

        rollout_prompts = train_questions[:rollout_batch_size]
        formatted_prompts = [r1_zero_prompt(q) for q in rollout_prompts]
        with torch.inference_mode():
            rollout_outputs = sample_rollouts(formatted_prompts, llm)

        repeated_ground_truths = [q for q in rollout_prompts for _ in range(group_size)]
        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_outputs,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization,
        )

        if loss_type == "grpo_clip":
            with torch.inference_mode():
                all_tokenized = []
                for i in range(n_microbatches_per_rollout_batch):
                    start = i * micro_train_batch_size
                    end = start + micro_train_batch_size
                    batch_prompts = rollout_prompts[start // group_size : (end - 1) // group_size + 1]
                    batch_outputs = rollout_outputs[start:end]
                    tokenized = tokenize_prompt_and_output(
                        [r1_zero_prompt(q) for q in batch_prompts],
                        batch_outputs,
                        tokenizer,
                    )
                    all_tokenized.append(tokenized)
                
                max_seq_len = max(t["input_ids"].size(1) for t in all_tokenized)
                
                old_log_probs = []
                for tokenized in all_tokenized:
                    input_ids = tokenized["input_ids"]
                    labels = tokenized["labels"]
                    pad_len = max_seq_len - input_ids.size(1)
                    if pad_len > 0:
                        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
                        labels = torch.nn.functional.pad(labels, (0, pad_len), value=tokenizer.pad_token_id)
                    
                    batch_old_log_probs = get_response_log_probs(
                        policy,
                        input_ids.to(device),
                        labels.to(device),
                    )["log_probs"].detach()
                    old_log_probs.append(batch_old_log_probs)
                old_log_probs = torch.cat(old_log_probs, dim=0)
        else:
            old_log_probs = None

        for _ in range(epochs_per_rollout_batch):
            for i in range(n_microbatches_per_rollout_batch):
                start = i * micro_train_batch_size
                end = start + micro_train_batch_size

                if start >= rollout_batch_size:
                    break

                end = min(end, rollout_batch_size)

                batch_prompts = rollout_prompts[start // group_size : (end - 1) // group_size + 1]
                batch_outputs = rollout_outputs[start:end]
                batch_advantages = advantages[start:end].to(device)
                batch_raw_rewards = raw_rewards[start:end].unsqueeze(1).to(device)
                batch_old_log_probs = old_log_probs[start:end].to(device) if old_log_probs is not None else None

                if len(batch_outputs) == 0:
                    continue

                tokenized = tokenize_prompt_and_output(
                    [r1_zero_prompt(q) for q in batch_prompts],
                    batch_outputs,
                    tokenizer,
                )
                input_ids = tokenized["input_ids"].to(device)
                labels = tokenized["labels"].to(device)
                response_mask = tokenized["response_mask"].to(device)

                if batch_old_log_probs is not None:
                    pad_len = batch_old_log_probs.size(1) - input_ids.size(1)
                    if pad_len > 0:
                        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
                        labels = torch.nn.functional.pad(labels, (0, pad_len), value=tokenizer.pad_token_id)
                        response_mask = torch.nn.functional.pad(response_mask, (0, pad_len), value=0)

                policy.train()
                res = get_response_log_probs(policy, input_ids, labels)
                policy_log_probs = res["log_probs"].to(device)

                ra = batch_raw_rewards if loss_type == "no_baseline" else None
                adv = batch_advantages if loss_type in {"reinforce_with_baseline", "grpo_clip"} else None

                print(policy_log_probs.shape)
                micro_loss, _ = grpo_microbatch_train_step(
                    policy_log_probs,
                    response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    raw_rewards=ra,
                    advantages=adv,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange,
                )

                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        if step % 10 == 0:
            val_reward = compute_validation_reward(policy, validation_questions, r1_zero_reward_fn, r1_zero_prompt)
            print(f"Step {step}: Validation Answer Reward = {val_reward:.4f}")

if __name__ == "__main__":
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    policy = AutoModelForCausalLM.from_pretrained(model_id).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch.manual_seed(42)

    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        r1_zero_prompt = f.read()

    train_data_path = "/data/a5-alignment/MATH/train.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"

    train_questions = []
    with open(train_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            train_questions.append(example["problem"])

    validation_questions = []
    with open(eval_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            validation_questions.append(example["problem"])

    n_grpo_steps = 100
    rollout_batch_size = 16
    group_size = 4
    sampling_temperature = 1.0
    sampling_min_tokens = 1
    sampling_max_tokens = 20
    epochs_per_rollout_batch = 1
    train_batch_size = 8
    gradient_accumulation_steps = 2
    gpu_memory_utilization = 0.8
    loss_type = "grpo_clip"
    use_std_normalization = True
    advantage_eps = 1e-6
    cliprange = 0.2
    learning_rate = 1e-5
    device = "cuda:0"
    seed = 42

    def format_prompt(question):
        return r1_zero_prompt.replace("{question}", question)

    grpo_train_loop(
        policy=policy,
        tokenizer=tokenizer,
        train_questions=train_questions,
        validation_questions=validation_questions,
        r1_zero_prompt=format_prompt,
        r1_zero_reward_fn=r1_zero_reward_fn,
        n_grpo_steps=n_grpo_steps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        sampling_temperature=sampling_temperature,
        sampling_min_tokens=sampling_min_tokens,
        sampling_max_tokens=sampling_max_tokens,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gpu_memory_utilization=gpu_memory_utilization,
        loss_type=loss_type,
        use_std_normalization=use_std_normalization,
        advantage_eps=advantage_eps,
        cliprange=cliprange,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )

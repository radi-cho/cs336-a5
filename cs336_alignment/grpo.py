import torch
from typing import List, Callable, Literal, Dict
from vllm import SamplingParams, LLM
from unittest.mock import patch
from transformers import PreTrainedModel
import wandb
import random
import gc

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
    train_answers: List[str],
    validation_data: List[tuple],
    r1_zero_prompt: Callable[[str], str],
    r1_zero_reward_fn: Callable[[str, str, str], Dict[str, float]],
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
    wandb_project: str = "cs336-grpo"
) -> None:
    wandb.init(project=wandb_project)

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

    llm = init_vllm(policy.config._name_or_path, device, seed, gpu_memory_utilization)

    def sample_rollouts(prompts: List[str], llm: LLM) -> List[str]:
        with torch.no_grad():
            vllm_outputs = llm.generate(prompts, sampling_params=sampling_params)
            outputs = []
            for vout in vllm_outputs:
                for out in vout.outputs:
                    outputs.append(out.text)
            return outputs

    def compute_validation_reward(model, validation_data: List[tuple], reward_fn: Callable[[str, str, str], Dict[str, float]], prompt_template: str, llm: LLM) -> float:
        load_policy_into_vllm_instance(model, llm)
        validation_data = random.sample(validation_data, min(1024, len(validation_data)))
        prompts = [prompt_template(q) for q, _ in validation_data]
        with torch.inference_mode():
            preds = sample_rollouts(prompts, llm)
        total_reward = 0.0
        total_accuracy = 0.0
        
        with open("validation_rollouts.txt", "a") as f:
            f.write(f"\n=== New Validation Run ===\n")
            for (q, s), o in zip(validation_data, preds):
                reward_dict = reward_fn(o, s)
                reward = reward_dict["reward"]
                answer_reward = reward_dict["answer_reward"]
                total_reward += reward
                total_accuracy += answer_reward
                f.write(f"Question: {q}\n")
                f.write(f"Ground Truth: {s}\n")
                f.write(f"Model Output: {o}\n")
                f.write(f"Reward: {reward:.4f}\n")
                f.write(f"Answer Accuracy: {answer_reward:.4f}\n")
                f.write("-" * 80 + "\n")

        avg_reward = total_reward / len(validation_data)
        avg_accuracy = total_accuracy / len(validation_data)
        return avg_reward, avg_accuracy

    for step in range(1, n_grpo_steps + 1):
        load_policy_into_vllm_instance(policy, llm)

        indices = random.sample(range(len(train_questions)), n_prompts_per_rollout_batch)
        rollout_prompts_unique = [train_questions[i] for i in indices]
        rollout_answers_unique = [train_answers[i] for i in indices]

        repeated_prompts = []
        for q in rollout_prompts_unique:
            formatted = r1_zero_prompt(q)
            repeated_prompts.extend([formatted] * group_size)

        with torch.inference_mode():
            rollout_outputs = sample_rollouts(repeated_prompts, llm)

        repeated_ground_truths = []
        for ans in rollout_answers_unique:
            repeated_ground_truths.extend([ans] * group_size)

        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_outputs,
            repeated_ground_truths,
            group_size,
            advantage_eps,
            use_std_normalization,
        )

        if loss_type == "grpo_clip":
            old_log_probs_list = []
            max_seq_len = 0
            all_tokenized = []
            for i in range(n_microbatches_per_rollout_batch):
                start = i * micro_train_batch_size
                end = min(start + micro_train_batch_size, rollout_batch_size)
                if start >= rollout_batch_size:
                    continue

                batch_indices = list(range(start, end))
                batch_prompts = [repeated_prompts[idx] for idx in batch_indices]
                batch_outputs = [rollout_outputs[idx] for idx in batch_indices]

                tokenized = tokenize_prompt_and_output(
                    batch_prompts,
                    batch_outputs,
                    tokenizer,
                )

                all_tokenized.append(tokenized)
                max_seq_len = max(max_seq_len, tokenized["input_ids"].size(1))

            for tokenized in all_tokenized:
                input_ids = tokenized["input_ids"]
                labels = tokenized["labels"]
                pad_len = max_seq_len - input_ids.size(1)
                if pad_len > 0:
                    input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
                    labels = torch.nn.functional.pad(labels, (0, pad_len), value=tokenizer.pad_token_id)

                with torch.inference_mode():
                    lp = get_response_log_probs(
                        policy,
                        input_ids.to(device),
                        labels.to(device),
                    )["log_probs"].detach()
                    old_log_probs_list.append(lp)

            del all_tokenized
            torch.cuda.empty_cache()
            gc.collect()
            all_tokenized = []

            old_log_probs = torch.cat(old_log_probs_list, dim=0)
        else:
            old_log_probs = None

        for _ in range(epochs_per_rollout_batch):
            loss = 0.0
            for i in range(n_microbatches_per_rollout_batch):
                start = i * micro_train_batch_size
                end = min(start + micro_train_batch_size, rollout_batch_size)
                if start >= rollout_batch_size:
                    break

                batch_indices = list(range(start, end))
                batch_size_actual = end - start
                if batch_size_actual == 0:
                    continue

                question_indices = [idx // group_size for idx in batch_indices]
                batch_prompts = [r1_zero_prompt(rollout_prompts_unique[qidx]) for qidx in question_indices]
                batch_outputs = [rollout_outputs[idx] for idx in batch_indices]
                batch_advantages = advantages[start:end].unsqueeze(1).to(device)
                batch_raw_rewards = raw_rewards[start:end].unsqueeze(1).to(device)

                tokenized = tokenize_prompt_and_output(
                    batch_prompts,
                    batch_outputs,
                    tokenizer,
                )
                input_ids = tokenized["input_ids"].to(device)
                labels = tokenized["labels"].to(device)
                response_mask = tokenized["response_mask"].to(device)

                if old_log_probs is not None:
                    pad_len = old_log_probs.size(1) - input_ids.size(1)
                    if pad_len > 0:
                        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=tokenizer.pad_token_id)
                        labels = torch.nn.functional.pad(labels, (0, pad_len), value=tokenizer.pad_token_id)
                        response_mask = torch.nn.functional.pad(response_mask, (0, pad_len), value=0)

                policy.train()
                # print(f"Max seq len: {input_ids.size(1)}")
                res = get_response_log_probs(policy, input_ids, labels)
                policy_log_probs = res["log_probs"].to(device)

                if old_log_probs is None:
                    batch_old_log_probs = policy_log_probs.detach()
                else:
                    batch_old_log_probs = old_log_probs[start:end].to(device)

                ra = batch_raw_rewards if loss_type == "no_baseline" else None
                adv = batch_advantages if loss_type in {"reinforce_with_baseline", "grpo_clip"} else None

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

                loss += micro_loss

                if (i + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"Step {step}, Loss: {loss.item():.4f}")
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/step": step,
                    })
                    loss = 0.0

                del input_ids, labels, response_mask, policy_log_probs, batch_old_log_probs
                if ra is not None:
                    del ra
                if adv is not None:
                    del adv
                torch.cuda.empty_cache()
                gc.collect()

        if step % 10 == 0:
            val_reward, val_accuracy = compute_validation_reward(policy, validation_data, r1_zero_reward_fn, r1_zero_prompt, llm)
            print(f"Step {step}: Validation Reward = {val_reward:.4f}, Validation Accuracy = {val_accuracy:.4f}")
            wandb.log({
                "validation/reward": val_reward,
                "validation/accuracy": val_accuracy,
                "validation/step": step,
            })

            torch.cuda.empty_cache()
            gc.collect()
            optimizer.zero_grad()

    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()


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
    random.seed(42)

    with open("cs336_alignment/prompts/r1_zero.prompt") as f:
        r1_zero_prompt_template = f.read()

    train_data_path = "/data/a5-alignment/MATH/train.jsonl"
    eval_data_path = "/data/a5-alignment/MATH/validation.jsonl"

    train_questions = []
    train_answers = []
    with open(train_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            train_questions.append(example["problem"])
            train_answers.append(example["answer"])

    validation_data = []
    with open(eval_data_path, "r") as f:
        for line in f:
            example = json.loads(line)
            validation_data.append((example["problem"], example["answer"]))

    n_grpo_steps = 200
    rollout_batch_size = 256
    group_size = 8
    sampling_temperature = 1.0
    sampling_min_tokens = 4
    sampling_max_tokens = 512
    epochs_per_rollout_batch = 1
    train_batch_size = 256
    gradient_accumulation_steps = 128
    gpu_memory_utilization = 0.2
    loss_type = "reinforce_with_baseline"
    use_std_normalization = False
    # use_std_normalization = True
    advantage_eps = 1e-6
    cliprange = 0.2
    learning_rate = 1e-5
    device = "cuda:0"
    seed = 42
    wandb_project = "cs336-grpo"

    def format_prompt(question):
        return r1_zero_prompt_template.replace("{question}", question)

    grpo_train_loop(
        policy=policy,
        tokenizer=tokenizer,
        train_questions=train_questions,
        train_answers=train_answers,
        validation_data=validation_data,
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
        wandb_project=wandb_project
    )

    grpo_train_loop(
        policy=policy,
        tokenizer=tokenizer,
        train_questions=train_questions,
        train_answers=train_answers,
        validation_data=validation_data,
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
        learning_rate=1e-6,
        device=device,
        seed=seed,
        wandb_project=wandb_project
    )

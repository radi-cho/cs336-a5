import torch
import numpy as np
from typing import List, Callable
from vllm import LLM, SamplingParams
from tqdm import tqdm
from cs336_alignment.entropy import compute_entropy


def log_generations(
    model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: SamplingParams,
    batch_size: int = 8,
) -> List[dict]:
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_truths = ground_truths[i:i + batch_size]
        outputs = model.generate(batch_prompts, sampling_params)

        for output, prompt, truth in zip(outputs, batch_prompts, batch_truths):
            response = output.outputs[0].text.strip()
            reward_info = reward_fn(truth, response)
            logits = output.outputs[0].logits if hasattr(output.outputs[0], "logits") else None
            avg_entropy = compute_entropy(torch.tensor(logits)).mean().item() if logits is not None else None
            result = {
                "prompt": prompt,
                "response": response,
                "ground_truth": truth,
                "reward": reward_info,
                "avg_token_entropy": avg_entropy,
                "response_length": len(response.split()),
            }
            results.append(result)

    response_lengths = [r["response_length"] for r in results]
    correct_responses = [r for r in results if r["reward"]["reward"] == 1.0]
    incorrect_responses = [r for r in results if r["reward"]["reward"] == 0.0]
    print("Logged Generations Summary:")
    print(f"Total: {len(results)}")
    print(f"Correct: {len(correct_responses)}")
    print(f"Incorrect: {len(incorrect_responses)}")
    print(f"Avg Length: {np.mean(response_lengths):.2f}")
    print(f"Avg Correct: {np.mean([r['response_length'] for r in correct_responses]) if correct_responses else 0:.2f}")
    print(f"Avg Incorrect: {np.mean([r['response_length'] for r in incorrect_responses]) if incorrect_responses else 0:.2f}")
    return results

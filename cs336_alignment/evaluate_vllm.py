import json
from tqdm import tqdm
from typing import Callable, List, Tuple
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts_solns: List[Tuple[str, str]],
    eval_sampling_params: SamplingParams
) -> None:
    prompts = [p for p, _ in prompts_solns]
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    for (prompt, solution), output in zip(prompts_solns, outputs):
        text = output.outputs[0].text
        score = reward_fn(text, solution)
        results.append({"prompt": prompt, "solution": solution, "pred": text, "score": score})
    return results


if __name__ == "__main__":
    with open("/data/a5-alignment/MATH/validation.jsonl") as fin:
        data = [json.loads(line) for line in fin]

    with open("cs336_alignment/prompts/r1_zero.prompt") as fin:
        tmpl = fin.read()

    prompts_solns = []
    for example in tqdm(data):
        prompt = tmpl.replace("{question}", example["problem"])
        prompts_solns.append((prompt, example["answer"]))

    # llm = LLM(model="/data/a5-alignment/models/Qwen2.5-Math-1.5B")
    llm = LLM(model="/home/c-radicho/cs336-a5/sft_outputs/size_128/temp_checkpoints/checkpoint_99")
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"])
    sampling_params.include_stop_str_in_output = True

    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts_solns, sampling_params)

    with open("math_zero_results.jsonl", "w") as fout:
        for r in results:
            fout.write(json.dumps(r) + "\n")

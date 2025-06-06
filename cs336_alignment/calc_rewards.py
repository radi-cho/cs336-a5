import json

results_path = "math_zero_results.jsonl"
correct_both = 0
correct_format_only = 0
incorrect_both = 0
total = 0

with open(results_path, "r") as fin:
    for line in fin:
        result = json.loads(line)
        score = result["score"]
        total += 1
        if score["format"] == 1 and score["answer"] == 1:
            correct_both += 1
        elif score["format"] == 1 and score["answer"] == 0:
            correct_format_only += 1
        elif score["format"] == 0 and score["answer"] == 0:
            incorrect_both += 1

print(f"(1) Correct (format=1, answer=1): {correct_both}")
print(f"(2) Format only (format=1, answer=0): {correct_format_only}")
print(f"(3) Incorrect format and answer (format=0, answer=0): {incorrect_both}")
print(f"Total evaluated: {total}")
print(f"Zero-shot baseline accuracy: {correct_both / total:.4f}")

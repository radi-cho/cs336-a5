import json

def analyze_math_results(file_path):
    count_format_and_answer_correct = 0
    count_format_correct_answer_incorrect = 0
    count_format_and_answer_incorrect = 0
    total = 0

    with open(file_path, "r") as fin:
        for line in fin:
            total += 1
            record = json.loads(line)
            score = record.get("score", {})
            fmt = score.get("format_reward", 0)
            ans = score.get("answer_reward", 0)

            if fmt == 1 and ans == 1:
                count_format_and_answer_correct += 1
                if count_format_and_answer_correct < 5:
                    print("11", record)
            elif fmt == 1 and ans == 0:
                count_format_correct_answer_incorrect += 1
                if count_format_correct_answer_incorrect < 5:
                    print("10", record)
            elif fmt == 0 and ans == 0:
                count_format_and_answer_incorrect += 1
                if count_format_and_answer_incorrect < 5:
                    print("00", record)

    accuracy = count_format_and_answer_correct / total if total > 0 else 0.0

    print(f"Total examples evaluated: {total}")
    print(f"1) format=1, answer=1, {count_format_and_answer_correct}")
    print(f"2) format=1, answer=0, {count_format_correct_answer_incorrect}")
    print(f"3) format=0, answer=0, {count_format_and_answer_incorrect}")
    print(f"accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    analyze_math_results("math_zero_results.jsonl")

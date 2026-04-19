#!/usr/bin/env python3
"""Convert generated questions from vllm_infer output to LlamaFactory format."""

import json
import argparse
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vllm_output(input_file: str) -> List[Dict]:
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} entries from {input_file}")
    return data


def extract_questions(vllm_data: List[Dict]) -> List[Dict]:
    questions_data = []
    for entry in vllm_data:
        predict = entry.get('predict', '').strip()
        if not predict:
            continue
        questions_data.append({'generated_question': predict})
    logger.info(f"Extracted {len(questions_data)} questions")
    return questions_data


def create_llamafactory_dataset(
    questions_data: List[Dict],
    output_file: str,
    dataset_type: str = "gsm8k"
):
    if dataset_type in ["gsm8k"]:
        task_instruction = (
            "You are given a grade school math word problem. "
            "Carefully read the problem and provide a step-by-step solution. "
            "Write the final numerical answer in the format: final answer: <answer>."
        )
    elif dataset_type == "math":
        task_instruction = (
            "You are given a competition-style mathematics problem. "
            "Provide a rigorous, step-by-step solution that clearly justifies each transformation. "
            "Present the final result inside a LaTeX boxed expression, i.e., write the answer as \\boxed{<answer>}."
        )
    elif dataset_type == "webinstruct":
        task_instruction = (
            "You are given a physics problem that requires numerical reasoning. "
            "Carefully read the problem and provide a step-by-step solution. "
            "Show all calculations and clearly explain your reasoning. "
            "Write the final answer in the format: final answer: <answer>."
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'gsm8k', 'math', or 'webinstruct'.")

    dataset = []
    for q_data in questions_data:
        dataset.append({
            "instruction": task_instruction,
            "input": q_data['generated_question'],
            "output": ""
        })

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved {len(dataset)} questions to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "webinstruct"])
    args = parser.parse_args()

    vllm_data = load_vllm_output(args.input_file)
    questions_data = extract_questions(vllm_data)

    if len(questions_data) == 0:
        raise ValueError("No valid questions extracted from input file")

    create_llamafactory_dataset(
        questions_data=questions_data,
        output_file=args.output_file,
        dataset_type=args.dataset_type
    )
    logger.info(f"Done. {len(questions_data)} questions saved to {args.output_file}")


if __name__ == "__main__":
    main()

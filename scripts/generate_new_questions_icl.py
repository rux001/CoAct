#!/usr/bin/env python3
"""Prepare ICL prompts for generating new questions from oracle-verified correct examples."""

import json
import random
import argparse
import logging
from typing import List, Dict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_oracle_data(oracle_file: str) -> List[Dict]:
    with open(oracle_file, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} oracle samples")
    return data


def extract_correct_questions(oracle_data: List[Dict]) -> List[str]:
    correct_questions = []
    for item in oracle_data:
        try:
            evaluation = item.get('evaluation', {})
            if not evaluation.get('response1_correct', False):
                continue
            question = item['question']
            if "### Response:\n" in question and "### Instruction:\n" in question:
                question = question.split("### Response:\n")[0]
                question = question.split("### Instruction:\n")[1]
                question = question.strip()
            else:
                question = question.strip()
            if question:
                correct_questions.append(question)
        except Exception as e:
            logger.warning(f"Error processing oracle sample: {e}")
            continue
    logger.info(f"Found {len(correct_questions)} correct questions")
    return correct_questions


def create_icl_prompt(few_shot_questions: List[str], dataset_type: str = "gsm8k") -> str:
    prompt_parts = []
    for question in few_shot_questions:
        prompt_parts.append(f"Q: {question}")

    if dataset_type in ["gsm8k", "math"]:
        generation_instruction = (
            "\nPrompt: Based on the examples above, generate ONE solvable math word problem "
            "with similar difficulty. Note that all the information needed to solve the problem "
            "should be included in the question. Output the question and nothing else.\nQ:"
        )
    elif dataset_type == "webinstruct":
        generation_instruction = (
            "\nPrompt: Based on the examples above, generate ONE solvable physics problem "
            "with similar difficulty and topic. The question should require numerical reasoning and may involve "
            "units, or currency. Ensure all information needed to solve the problem is included. "
            "Output the question and nothing else.\nQ:"
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'gsm8k', 'math', or 'webinstruct'.")

    prompt_parts.append(generation_instruction)
    return "\n".join(prompt_parts)


def create_icl_prompts(
    correct_questions: List[str],
    num_prompts: int = 2000,
    num_few_shot: int = 3,
    dataset_type: str = "gsm8k"
) -> List[str]:
    if len(correct_questions) < num_few_shot:
        raise ValueError(f"Need at least {num_few_shot} correct questions for ICL")

    prompts = []
    for _ in tqdm(range(num_prompts), desc="Creating prompts"):
        few_shot = random.sample(correct_questions, num_few_shot)
        prompt = create_icl_prompt(few_shot, dataset_type=dataset_type)
        prompts.append(prompt)
    return prompts


def save_prompts_for_vllm_infer(prompts: List[str], output_file: str):
    sft_data = [{"instruction": p, "input": "", "output": ""} for p in prompts]
    with open(output_file, 'w') as f:
        json.dump(sft_data, f, indent=2)
    logger.info(f"Saved {len(sft_data)} prompts to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="icl_prompts_for_vllm.json")
    parser.add_argument("--num_prompts", type=int, default=2000)
    parser.add_argument("--num_few_shot", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_type", type=str, default="gsm8k",
                        choices=["gsm8k", "math", "webinstruct"])
    args = parser.parse_args()

    random.seed(args.seed)
    oracle_data = load_oracle_data(args.oracle_file)
    correct_questions = extract_correct_questions(oracle_data)

    if len(correct_questions) < args.num_few_shot:
        raise ValueError(
            f"Not enough correct questions ({len(correct_questions)}) "
            f"for {args.num_few_shot}-shot ICL"
        )

    icl_prompts = create_icl_prompts(
        correct_questions=correct_questions,
        num_prompts=args.num_prompts,
        num_few_shot=args.num_few_shot,
        dataset_type=args.dataset_type
    )
    save_prompts_for_vllm_infer(icl_prompts, args.output_file)
    logger.info(f"Done. {len(icl_prompts)} prompts saved to {args.output_file}")


if __name__ == "__main__":
    main()

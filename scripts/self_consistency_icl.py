#!/usr/bin/env python3
"""Calculate self-consistency scores for ICL-generated questions and their responses."""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
from multiprocessing import Pool, cpu_count

from paths import get_paths
from utils import (
    extract_answer as _extract_answer_by_prefix,
    get_task_instruction as _get_task_instruction_by_prefix,
)

DATASET_PREFIX = "gsm8k"


def set_dataset_prefix(prefix: str) -> None:
    global DATASET_PREFIX
    DATASET_PREFIX = (prefix or "gsm8k").lower()


def extract_final_answer(text: str) -> Optional[str]:
    return _extract_answer_by_prefix(text, DATASET_PREFIX)


def get_task_instruction() -> str:
    return _get_task_instruction_by_prefix(DATASET_PREFIX)


def load_responses(response_files: List[str]) -> Dict[int, List[Dict]]:
    questions_responses = {}
    for file_path in response_files:
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                prompt = entry.get('prompt', '')
                if 'idx' not in entry:
                    continue
                idx = entry['idx']
                response = entry.get('predict', '')
                if idx not in questions_responses:
                    questions_responses[idx] = []
                questions_responses[idx].append({'response': response, 'prompt': prompt})
    return questions_responses


def load_responses_from_jsonl_files(response_files: List[str], questions_file: Optional[str] = None) -> List[Dict]:
    all_responses = []

    questions_list = []
    if questions_file and Path(questions_file).exists():
        with open(questions_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                questions_list.append(entry.get('predict', '').strip())

    with open(response_files[0], 'r') as f:
        for line_idx, line in enumerate(f):
            entry = json.loads(line)
            if line_idx < len(questions_list):
                question = questions_list[line_idx]
            else:
                prompt = entry.get('prompt', '')
                question = ""
                if '### Instruction:' in prompt:
                    parts = prompt.split('### Instruction:')
                    if len(parts) > 1:
                        instruction_part = parts[1]
                        for delimiter in ['### Input:', '\n\n', '### Response:']:
                            if delimiter in instruction_part:
                                question = instruction_part.split(delimiter)[0].strip()
                                if question.startswith('Question:'):
                                    question = question[9:].strip()
                                break
            all_responses.append({'question': question, 'responses': [entry.get('predict', '')]})

    for file_path in response_files[1:]:
        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                entry = json.loads(line)
                if line_idx < len(all_responses):
                    all_responses[line_idx]['responses'].append(entry.get('predict', ''))

    return all_responses


def calculate_self_consistency(responses: List[str]) -> Tuple[int, Dict[str, int]]:
    answers = [a for a in (extract_final_answer(r) for r in responses) if a is not None]
    if not answers:
        return 0, {}
    answer_counts = Counter(answers)
    return answer_counts.most_common(1)[0][1], dict(answer_counts)


def _process_single_question(args):
    question, responses = args
    if len(responses) < 2:
        return None

    extracted_answers = [(r, a) for r, a in ((r, extract_final_answer(r)) for r in responses) if a is not None]
    if not extracted_answers:
        return None

    answers_only = [a for _, a in extracted_answers]
    answer_counts = Counter(answers_only)
    if len(answer_counts) == 1:
        return None

    sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
    most_voted_answer = sorted_answers[0][0]
    least_voted_answer = sorted_answers[-1][0]

    most_voted_responses = [r for r, a in extracted_answers if a == most_voted_answer]
    least_voted_responses = [r for r, a in extracted_answers if a == least_voted_answer]
    if not most_voted_responses or not least_voted_responses:
        return None

    task_instruction = get_task_instruction()
    return {
        'conversations': [{'from': 'human', 'value': task_instruction + "Question: " + question}],
        'chosen': {'from': 'gpt', 'value': most_voted_responses[-1]},
        'rejected': {'from': 'gpt', 'value': min(least_voted_responses, key=len)},
        'consistency_score': answer_counts.most_common(1)[0][1],
        'answer_counts': dict(answer_counts),
        'most_voted_answer': most_voted_answer,
        'least_voted_answer': least_voted_answer,
        'num_responses': len(responses),
        'num_chosen': len(most_voted_responses),
        'num_rejected': len(least_voted_responses),
    }


def create_preference_pairs(questions_with_responses: List[Dict], consistency_threshold: int = 3,
                            num_workers: int = None) -> Tuple[List[Dict], List[Dict]]:
    total_questions = len(questions_with_responses)
    if num_workers is None:
        num_workers = max(1, min(cpu_count() - 1, 32))

    args_list = [(item['question'], item['responses']) for item in questions_with_responses]
    chunk_size = max(10, total_questions // (num_workers * 4))

    start_time = time.time()
    high_consistency_pairs: List[Dict] = []
    all_pairs: List[Dict] = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_process_single_question, args_list, chunksize=chunk_size):
            if result is None:
                continue
            all_pairs.append(result)
            if result['consistency_score'] >= consistency_threshold:
                high_consistency_pairs.append(result)

    elapsed = time.time() - start_time
    print(f"Processed {total_questions} questions in {elapsed:.1f}s ({num_workers} workers): {len(all_pairs)} pairs, {len(high_consistency_pairs)} high-consistency")
    return high_consistency_pairs, all_pairs


def save_results(high_consistency_pairs: List[Dict], all_pairs: List[Dict], paths):
    high_consistency_llamafactory = [
        {'conversations': p['conversations'], 'chosen': p['chosen'], 'rejected': p['rejected']}
        for p in high_consistency_pairs
    ]

    with open(paths.icl_high_consistent(), 'w') as f:
        json.dump(high_consistency_llamafactory, f, indent=2)

    with open(paths.icl_all_with_scores(), 'w') as f:
        json.dump(all_pairs, f, indent=2)

    if all_pairs:
        consistency_scores = [p['consistency_score'] for p in all_pairs]
        score_distribution = Counter(consistency_scores)
        print(
            f"Saved {len(high_consistency_llamafactory)} high-consistency pairs "
            f"and {len(all_pairs)} total pairs; "
            f"mean_score={sum(consistency_scores)/len(consistency_scores):.2f} "
            f"distribution={dict(sorted(score_distribution.items()))}"
        )
    else:
        print("Saved 0 pairs (empty input)")


def main():
    parser = argparse.ArgumentParser(description='Calculate self-consistency for ICL-generated responses')
    parser.add_argument('--response_files', nargs='+', required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--dataset-prefix', type=str, required=True, choices=['gsm8k', 'math', 'webinstruct'])
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--consistency_threshold', type=int, default=4)
    parser.add_argument('--questions_file', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)

    args = parser.parse_args()
    set_dataset_prefix(args.dataset_prefix)

    paths = get_paths(args.output_root, args.dataset_prefix, args.iteration)
    paths.makedirs()

    print(f"self_consistency_icl: dataset={args.dataset_prefix} threshold={args.consistency_threshold} files={len(args.response_files)}")

    questions_with_responses = load_responses_from_jsonl_files(args.response_files, questions_file=args.questions_file)
    print(f"Loaded {len(questions_with_responses)} questions with {len(args.response_files)} responses each")

    high_consistency_pairs, all_pairs = create_preference_pairs(
        questions_with_responses, args.consistency_threshold, num_workers=args.num_workers,
    )
    save_results(high_consistency_pairs, all_pairs, paths)


if __name__ == "__main__":
    main()

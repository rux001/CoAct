#!/usr/bin/env python3
"""Construct preferred/dispreferred pairs from generated responses with consistency scoring."""

import json
import os
import glob
import time
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from multiprocessing import Pool, cpu_count

from knn import KNNErrorDetector
from paths import get_paths
from utils import (
    extract_answer as _extract_answer_by_prefix,
    extract_question_from_prompt,
)

DATASET_PREFIX = "gsm8k"


def set_dataset_prefix(prefix: str) -> None:
    global DATASET_PREFIX
    DATASET_PREFIX = (prefix or "gsm8k").lower()


def extract_answer(text: str) -> Optional[str]:
    return _extract_answer_by_prefix(text, DATASET_PREFIX)


def load_k_responses(file_path: str) -> Dict[str, List[str]]:
    prompt_responses = defaultdict(list)
    with open(file_path, 'r') as f:
        data = json.load(f)
    for item in data:
        prompt = item['original_sample']['instruction']
        for response_item in item.get('responses', []):
            prompt_responses[prompt].append(response_item.get('response', ''))
    return dict(prompt_responses)


def load_jsonl_responses(file_path: str) -> Dict[str, tuple]:
    prompt_data = defaultdict(lambda: ([], None))
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompt, response, label = data['prompt'], data['predict'], data.get('label', '')
            if prompt not in prompt_data or prompt_data[prompt][1] is None:
                prompt_data[prompt] = ([response], label)
            else:
                prompt_data[prompt][0].append(response)
    return dict(prompt_data)


def _construct_single_preference_pair(args):
    prompt, responses, label = args
    if not responses:
        return None

    extracted_answers = [extract_answer(r) for r in responses]
    extracted_answers = [a for a in extracted_answers if a is not None]
    if not extracted_answers:
        return None

    answer_counts = Counter(extracted_answers)
    if len(answer_counts) == 1:
        return None

    sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
    most_voted_answer, _ = sorted_answers[0]
    least_voted_answer, _ = sorted_answers[-1]

    most_voted_responses = [r for r, a in zip(responses, [extract_answer(r) for r in responses]) if a == most_voted_answer]
    least_voted_responses = [r for r, a in zip(responses, [extract_answer(r) for r in responses]) if a == least_voted_answer]

    ground_truth = None
    ground_truth_full_response = None
    accuracy = None
    if label:
        ground_truth_full_response = label
        ground_truth = extract_answer(label)
        if ground_truth:
            accuracy = 1 if most_voted_answer == ground_truth else 0

    return {
        'prompt': prompt,
        'positive_response': most_voted_responses[-1],
        'negative_response': min(least_voted_responses, key=len),
        'consistency_score': answer_counts.most_common(1)[0][1],
        'answer_counts': dict(answer_counts),
        'most_voted_answer': most_voted_answer,
        'least_voted_answer': least_voted_answer,
        'ground_truth': ground_truth,
        'ground_truth_full_response': ground_truth_full_response,
        'accuracy': accuracy,
    }


def construct_preference_pairs(prompt_responses: Dict[str, List[str]], prompt_labels: Dict[str, str] = None, num_workers: int = None) -> List[Dict[str, Any]]:
    total_prompts = len(prompt_responses)
    if num_workers is None:
        num_workers = max(1, min(cpu_count() - 1, 32))

    args_list = [
        (prompt, responses, prompt_labels.get(prompt) if prompt_labels else None)
        for prompt, responses in prompt_responses.items()
    ]
    chunk_size = max(10, total_prompts // (num_workers * 4))

    start_time = time.time()
    preference_pairs = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_construct_single_preference_pair, args_list, chunksize=chunk_size):
            if result is not None:
                preference_pairs.append(result)

    elapsed = time.time() - start_time
    print(f"Constructed {len(preference_pairs)} preference pairs from {total_prompts} prompts in {elapsed:.1f}s ({num_workers} workers)")
    return preference_pairs


def _process_single_question(args):
    prompt, responses, label = args
    if not responses:
        return {
            'prompt': prompt, 'consistency_score': 0, 'answer_counts': {},
            'num_unique_answers': 0, 'ground_truth': None,
            'ground_truth_full_response': None, 'accuracy': None,
        }

    extracted_answers = [a for a in (extract_answer(r) for r in responses) if a is not None]
    answer_counts = Counter(extracted_answers)
    total = len(extracted_answers)
    consistency_score = answer_counts.most_common(1)[0][1] if total > 0 and answer_counts else 0

    ground_truth = None
    ground_truth_full_response = None
    accuracy = None
    if label:
        ground_truth_full_response = label
        ground_truth = extract_answer(label)
        if ground_truth and answer_counts:
            most_voted = max(answer_counts.keys(), key=lambda x: answer_counts[x])
            accuracy = 1 if most_voted == ground_truth else 0

    return {
        'prompt': prompt, 'consistency_score': consistency_score,
        'answer_counts': dict(answer_counts), 'num_unique_answers': len(answer_counts),
        'ground_truth': ground_truth, 'ground_truth_full_response': ground_truth_full_response,
        'accuracy': accuracy,
    }


def analyze_all_questions(prompt_responses: Dict[str, List[str]], prompt_labels: Dict[str, str] = None, num_workers: int = None) -> List[Any]:
    total_prompts = len(prompt_responses)
    if num_workers is None:
        num_workers = max(1, min(cpu_count() - 1, 32))

    args_list = [
        (prompt, responses, prompt_labels.get(prompt) if prompt_labels else None)
        for prompt, responses in prompt_responses.items()
    ]
    chunk_size = max(10, total_prompts // (num_workers * 4))

    start_time = time.time()
    all_questions = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_process_single_question, args_list, chunksize=chunk_size):
            all_questions.append(result)

    elapsed = time.time() - start_time
    print(f"Analyzed {total_prompts} questions in {elapsed:.1f}s ({num_workers} workers)")
    return all_questions


def filter_high_consistency(pairs: List[Dict[str, Any]], min_score: int = 3) -> List[Dict[str, Any]]:
    return [pair for pair in pairs if pair['consistency_score'] >= min_score]


def select_least_consistent(pairs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return sorted(pairs, key=lambda x: x['consistency_score'])[:n]


def select_most_consistent(pairs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return sorted(pairs, key=lambda x: x['consistency_score'], reverse=True)[:n]


def load_reference_set(reference_file: str) -> Optional[List[Dict[str, Any]]]:
    try:
        with open(reference_file, 'r') as f:
            reference_data = json.load(f)
        print(f"Loaded {len(reference_data)} reference samples from {reference_file}")
        return reference_data
    except Exception as e:
        print(f"Error loading reference set: {e}")
        return None


def select_ood_samples_with_knn(candidate_pairs, reference_set, model, tokenizer, device,
                                n_select, k_neighbors=50, batch_size=8):
    print(f"KNN OOD selection: reference={len(reference_set)} candidates={len(candidate_pairs)} selecting top {n_select}")

    detector = KNNErrorDetector(model=model, tokenizer=tokenizer, k=k_neighbors, device=device, max_length=2048)
    detector.build_reference_sets_from_oracle_evaluation(reference_set, batch_size=batch_size)

    candidate_questions = [extract_question_from_prompt(pair["prompt"]) for pair in candidate_pairs]
    knn_scores = detector.compute_knn_scores_batch(candidate_questions, batch_size=batch_size)

    pairs_with_scores = []
    for pair, scores in zip(candidate_pairs, knn_scores):
        scored = pair.copy()
        scored["ood_score"] = float(scores["ood_score"])
        scored["id_score"] = float(scores["id_score"])
        scored["r_correct"] = float(scores["r_correct"])
        pairs_with_scores.append(scored)

    selected_pairs = sorted(pairs_with_scores, key=lambda x: x["ood_score"], reverse=True)[:n_select]
    if selected_pairs:
        scores = [p["ood_score"] for p in selected_pairs]
        print(f"Selected {len(selected_pairs)} OOD pairs (score range {min(scores):.4f} - {max(scores):.4f}, avg {sum(scores)/len(scores):.4f})")
    return selected_pairs


def create_llama_factory_format(preference_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    llama_data = []
    for pair in preference_pairs:
        prompt_lines = pair['prompt'].split('\n')
        question = pair['prompt']
        for line in prompt_lines:
            if line.startswith('Question:'):
                question = line.replace('Question:', '').strip()
                break
        llama_data.append({
            "conversations": [{"from": "human", "value": question}],
            "chosen": {"from": "gpt", "value": pair['positive_response']},
            "rejected": {"from": "gpt", "value": pair['negative_response']},
        })
    return llama_data


def analyze_consistency_accuracy(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_pairs = [p for p in pairs if p.get('accuracy') is not None]
    if not valid_pairs:
        return {"error": "No pairs with accuracy information found"}

    total_accuracy = sum(p['accuracy'] for p in valid_pairs) / len(valid_pairs)
    consistency_groups = defaultdict(list)
    for pair in valid_pairs:
        consistency_groups[pair['consistency_score']].append(pair['accuracy'])

    accuracy_by_consistency = {score: sum(accs) / len(accs) for score, accs in consistency_groups.items()}
    consistency_scores = [p['consistency_score'] for p in valid_pairs]
    accuracies = [p['accuracy'] for p in valid_pairs]

    n = len(valid_pairs)
    mean_c = sum(consistency_scores) / n
    mean_a = sum(accuracies) / n
    numerator = sum((consistency_scores[i] - mean_c) * (accuracies[i] - mean_a) for i in range(n))
    denominator = (sum((c - mean_c) ** 2 for c in consistency_scores) * sum((a - mean_a) ** 2 for a in accuracies)) ** 0.5
    correlation = numerator / denominator if denominator != 0 else 0

    return {
        "total_pairs_with_accuracy": len(valid_pairs),
        "overall_accuracy": total_accuracy,
        "accuracy_by_consistency": accuracy_by_consistency,
        "consistency_accuracy_correlation": correlation,
        "consistency_score_distribution": dict(Counter(consistency_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description='Construct preference pairs from generated responses')
    parser.add_argument('--dataset', type=str, default='it0')
    parser.add_argument('--dataset-prefix', type=str, default='gsm8k', choices=['gsm8k', 'math', 'webinstruct'])
    parser.add_argument('--k_responses_file', type=str, default=None)
    parser.add_argument('--jsonl_files_pattern', type=str, default=None)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--high_consistency_threshold', type=int, default=3)
    parser.add_argument('--n_high_consistency', type=int, default=150)
    parser.add_argument('--n_oracle', type=int, default=150)
    parser.add_argument('--llama_factory_output_dir', type=str, required=True)
    parser.add_argument('--generated_responses_dir', type=str, default=None)
    parser.add_argument('--oracle_evaluations_dir', type=str, default=None)
    parser.add_argument('--merged_lora_root', type=str, default=None)
    parser.add_argument('--iteration', type=int, default=0, help='Current iteration number')
    parser.add_argument('--use_knn_selection', action='store_true')
    parser.add_argument('--reference_set_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--k_neighbors', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    dataset_prefix = args.dataset_prefix.lower()
    set_dataset_prefix(dataset_prefix)

    dataset_iteration = 0
    if args.dataset.startswith('it'):
        try:
            dataset_iteration = int(args.dataset[2:])
        except ValueError:
            dataset_iteration = 0
    prev_iteration = max(dataset_iteration - 1, 0)

    paths = get_paths(args.output_root, dataset_prefix, dataset_iteration)
    paths.makedirs()

    if args.generated_responses_dir is not None:
        if not args.k_responses_file:
            args.k_responses_file = os.path.join(
                args.generated_responses_dir,
                f'{dataset_prefix}_al_{args.dataset}_k8_responses.json',
            )
        if not args.jsonl_files_pattern:
            args.jsonl_files_pattern = os.path.join(
                args.generated_responses_dir,
                f'{dataset_prefix}_al_{args.dataset}_run*.jsonl',
            )

    if args.use_knn_selection:
        prev_paths = get_paths(args.output_root, dataset_prefix, prev_iteration)
        if not args.reference_set_path:
            args.reference_set_path = prev_paths.eval_high_conf()
        if not args.model_path:
            args.model_path = prev_paths.merged_model()

    os.makedirs(args.llama_factory_output_dir, exist_ok=True)
    print(f"self_consistency: dataset={dataset_prefix} iter={args.dataset}")

    if os.path.exists(args.k_responses_file):
        k_responses = load_k_responses(args.k_responses_file)
        print(f"Loaded {len(k_responses)} prompts from k_responses")
    else:
        print(f"Warning: k_responses file not found: {args.k_responses_file}")
        k_responses = {}

    all_prompt_responses = k_responses.copy()
    all_prompt_labels: Dict[str, str] = {}

    jsonl_files = glob.glob(args.jsonl_files_pattern)
    for jsonl_file in jsonl_files:
        jsonl_data = load_jsonl_responses(jsonl_file)
        for prompt, (responses, label) in jsonl_data.items():
            if not k_responses:
                if prompt in all_prompt_responses:
                    all_prompt_responses[prompt].extend(responses)
                else:
                    all_prompt_responses[prompt] = responses
            if label and prompt not in all_prompt_labels:
                all_prompt_labels[prompt] = label

    total_responses = sum(len(r) for r in all_prompt_responses.values())
    print(f"Loaded {len(all_prompt_responses)} prompts / {total_responses} responses from {len(jsonl_files)} JSONL files")

    all_questions = analyze_all_questions(all_prompt_responses, all_prompt_labels, num_workers=args.num_workers)
    consistency_scores = [q['consistency_score'] for q in all_questions]
    if consistency_scores:
        score_distribution = Counter(consistency_scores)
        print(f"Consistency scores: distribution={dict(sorted(score_distribution.items()))} mean={sum(consistency_scores)/len(consistency_scores):.2f}")

    preference_pairs = construct_preference_pairs(all_prompt_responses, all_prompt_labels, num_workers=args.num_workers)

    accuracy_analysis = analyze_consistency_accuracy(preference_pairs)
    if not accuracy_analysis.get('error'):
        print(f"Accuracy: overall={accuracy_analysis['overall_accuracy']:.3f} correlation={accuracy_analysis['consistency_accuracy_correlation']:.3f}")

    def _dump_json(path, obj):
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    _dump_json(paths.all_pairs(), preference_pairs)

    high_consistency_pairs = filter_high_consistency(preference_pairs, min_score=args.high_consistency_threshold)
    print(f"High-consistency pairs (score >= {args.high_consistency_threshold}): {len(high_consistency_pairs)}")

    least_consistent_high: List[Dict[str, Any]] = []
    if len(high_consistency_pairs) >= args.n_high_consistency:
        if args.use_knn_selection:
            reference_set = load_reference_set(args.reference_set_path)
            if not reference_set:
                print("Reference set empty/missing; falling back to consistency-based selection")
                least_consistent_high = select_least_consistent(high_consistency_pairs, args.n_high_consistency)
            else:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM

                tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.eval()

                least_consistent_high = select_ood_samples_with_knn(
                    high_consistency_pairs, reference_set, model, tokenizer, args.device,
                    args.n_high_consistency, args.k_neighbors, args.batch_size,
                )
                del model, tokenizer
                torch.cuda.empty_cache()
        else:
            least_consistent_high = select_least_consistent(high_consistency_pairs, args.n_high_consistency)

        _dump_json(paths.high_conf_candidates(), least_consistent_high)
        _dump_json(
            os.path.join(args.llama_factory_output_dir, f'{dataset_prefix}_least_consistent_high_{len(least_consistent_high)}.json'),
            create_llama_factory_format(least_consistent_high),
        )

    oracle_pairs = None
    if len(preference_pairs) >= args.n_oracle:
        oracle_pairs = select_least_consistent(preference_pairs, args.n_oracle)
        _dump_json(paths.oracle_candidates(), oracle_pairs)
        _dump_json(
            os.path.join(args.llama_factory_output_dir, f'{dataset_prefix}_oracle_{len(oracle_pairs)}.json'),
            create_llama_factory_format(oracle_pairs),
        )

    oracle_selected_prompts = {pair['prompt'] for pair in least_consistent_high}
    if oracle_pairs is not None:
        oracle_selected_prompts.update(pair['prompt'] for pair in oracle_pairs)
    self_label_pairs = [pair for pair in high_consistency_pairs if pair['prompt'] not in oracle_selected_prompts]
    _dump_json(paths.self_label(), self_label_pairs)
    _dump_json(
        os.path.join(args.llama_factory_output_dir, f'{dataset_prefix}_self_label_{len(self_label_pairs)}.json'),
        create_llama_factory_format(self_label_pairs),
    )

    print(
        f"Done: analyzed={len(all_questions)} pairs={len(preference_pairs)} "
        f"high_consistency={len(high_consistency_pairs)} "
        f"oracle_high_cons={len(least_consistent_high)} "
        f"oracle_general={len(oracle_pairs) if oracle_pairs else 0} "
        f"self_label={len(self_label_pairs)}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Oracle evaluation using GPT-5 for preference pairs. Supports GSM8K, MATH, WebInstruct."""

import json
import os
import argparse
import requests
import time
import random
from typing import Dict, List, Any

from paths import get_paths
from utils import extract_question_from_prompt

DATASET_TYPE = "gsm8k"


def set_dataset_type(dt: str) -> None:
    global DATASET_TYPE
    DATASET_TYPE = dt.lower()


def setup_azure_client(api_key: str, endpoint: str, api_version: str) -> Dict[str, str]:
    return {'api_key': api_key, 'endpoint': endpoint, 'api_version': api_version}


def create_evaluation_prompt(question: str, response1: str, response2: str,
                             ground_truth_full_response: str = None) -> str:
    if DATASET_TYPE == "webinstruct":
        prompt = f"""You are an expert evaluator for scientific and academic questions requiring numerical reasoning. Evaluate two responses to a question and output your evaluation as a JSON object.

Question: {question}

Response 1: {response1}

Response 2: {response2}

Given the ground truth full response: {ground_truth_full_response}

Evaluate the responses following this logic:
1. Check if Response 1's final numerical answer is correct (consider scientific notation, units, and currency)
2. Check if Response 2's final numerical answer is correct (consider scientific notation, units, and currency)
3. Determine preference based on correctness:
   - If only one response is correct, prefer the correct one
   - If both responses are correct, prefer the one with better reasoning/explanation
   - If both responses are incorrect, prefer the one with better reasoning/explanation

Note: When comparing numerical answers, consider:
- Scientific notation (e.g., 1.4 x 10^9 = 1400000000)
- Currency formats (e.g., $8,645.09)
- Units (e.g., 17.16 m/s - the numerical value matters)
- Reasonable rounding differences

Output your evaluation as a JSON object with the following structure:
{{
    "response1_correct": true/false,
    "response2_correct": true/false,
    "response1_preferred": true/false,
    "reasoning": "Brief explanation of your evaluation"
}}

Only output the JSON object, no additional text."""
    else:
        prompt = f"""You are an expert evaluator for mathematical reasoning problems. Evaluate two responses to a math question and output your evaluation as a JSON object.

Question: {question}

Response 1: {response1}

Response 2: {response2}

Given the ground truth full response: {ground_truth_full_response}
Evaluate the responses following this logic:
1. Check if Response 1's final answer is correct
2. Check if Response 2's final answer is correct
3. Determine preference based on correctness:
   - If only one response is correct, prefer the correct one
   - If both responses are correct, prefer the one with better reasoning/explanation
   - If both responses are incorrect, prefer the one with better reasoning/explanation

Output your evaluation as a JSON object with the following structure:
{{
    "response1_correct": true/false,
    "response2_correct": true/false,
    "response1_preferred": true/false,
    "reasoning": "Brief explanation of your evaluation"
}}

Only output the JSON object, no additional text."""
    return prompt


def generate_correct_response(client_config: Dict[str, str], deployment_name: str,
                              question: str, correct_answer: str = None, max_retries: int = 3) -> str:
    if DATASET_TYPE == "webinstruct":
        prompt = f"""Provide a step-by-step reasoning process for this scientific or academic question and then write the final numerical answer on a new line in the format: final answer: <answer>.

Question: {question}

Please provide:
1. Step-by-step reasoning showing all calculations
2. The correct final numerical answer (include units if applicable)
3. Use scientific notation if appropriate (e.g., 1.4 x 10^9)
4. Include currency symbols if relevant (e.g., $8,645.09)
5. Include units if applicable (e.g., 17.16 m/s)

Make sure your solution is accurate, clearly explained, and includes proper units."""
        system_content = "You are an expert in scientific and academic reasoning. Provide accurate, step-by-step solutions with proper units and notation."
    else:
        prompt = f"""Provide a step-by-step reasoning process and then write the final numerical answer on a new line in the format: final answer: <answer>.

Question: {question}

Please provide:
1. Step-by-step reasoning
2. The correct final answer

Make sure your solution is accurate and clearly explained."""
        system_content = "You are an expert in mathematical reasoning. Provide accurate, step-by-step solutions."

    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json', 'api-key': client_config['api_key']})

    for attempt in range(max_retries):
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
            response = session.post(client_config['endpoint'], json=payload, timeout=60)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"Error generating correct response: {e}"


def evaluate_preference_pair(client_config: Dict[str, str], deployment_name: str,
                             question: str, response1: str, response2: str,
                             ground_truth_full_response: str = None, max_retries: int = 3) -> Dict[str, Any]:
    prompt = create_evaluation_prompt(question, response1, response2, ground_truth_full_response)

    if DATASET_TYPE == "webinstruct":
        system_message = "You are an expert evaluator for scientific and academic questions requiring numerical reasoning. Be precise and thorough in your evaluations, considering scientific notation, units, and currency formats."
    else:
        system_message = "You are an expert evaluator for mathematical reasoning problems. Be precise and thorough in your evaluations."

    session = requests.Session()
    session.headers.update({'Content-Type': 'application/json', 'api-key': client_config['api_key']})

    for attempt in range(max_retries):
        try:
            payload = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 0.7
            }
            response = session.post(client_config['endpoint'], json=payload, timeout=60)
            response.raise_for_status()
            evaluation_text = response.json()['choices'][0]['message']['content'].strip()
            parsed = parse_evaluation(evaluation_text)
            parsed['raw_evaluation'] = evaluation_text
            return parsed
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    'error': str(e),
                    'correct_answer': 'Error',
                    'better_reasoning': 'Error',
                    'overall_preference': 'Error',
                    'raw_evaluation': 'Failed to get evaluation'
                }


def parse_evaluation(evaluation_text: str) -> Dict[str, Any]:
    try:
        ej = json.loads(evaluation_text.strip())
        return {
            'response1_correct': ej.get('response1_correct', False),
            'response2_correct': ej.get('response2_correct', False),
            'response1_preferred': ej.get('response1_preferred', False),
            'reasoning': ej.get('reasoning', '')
        }
    except json.JSONDecodeError:
        lines = evaluation_text.split('\n')
        result = {
            'response1_correct': None, 'response2_correct': None,
            'response1_preferred': None, 'reasoning': '', 'correct_answer': None
        }
        for line in lines:
            line = line.strip()
            if line.startswith('Response 1 Correct:'):
                result['response1_correct'] = line.replace('Response 1 Correct:', '').strip().lower() == 'yes'
            elif line.startswith('Response 2 Correct:'):
                result['response2_correct'] = line.replace('Response 2 Correct:', '').strip().lower() == 'yes'
            elif line.startswith('Response 1 Preferred:'):
                result['response1_preferred'] = line.replace('Response 1 Preferred:', '').strip().lower() == 'yes'
        return result


def determine_preference_correctness(evaluation: Dict[str, Any],
                                     original_positive: str, original_negative: str,
                                     ground_truth_full_response: str = None,
                                     correct_response: str = None) -> Dict[str, Any]:
    r1c = evaluation['response1_correct']
    r2c = evaluation.get('response2_correct', False)
    r1p = evaluation['response1_preferred']
    reasoning = evaluation.get('reasoning', '')

    if r1p is True:
        preference_correct = True
    elif r1p is False:
        preference_correct = False
    else:
        preference_correct = None

    if r1c is True and r2c is False:
        preference_logic = "correct_vs_incorrect"
    elif r1c is False and r2c is True:
        preference_logic = "incorrect_vs_correct"
    elif r1c is True and r2c is True:
        preference_logic = "both_correct"
    elif r1c is False and r2c is False:
        preference_logic = "both_incorrect"
        if correct_response:
            preference_correct = False
            if ground_truth_full_response and correct_response == ground_truth_full_response:
                preference_logic = "both_incorrect_ground_truth_corrected"
            else:
                preference_logic = "both_incorrect_gpt_corrected"
    else:
        preference_logic = "unknown_logic"

    result = {
        'response1_correct': r1c, 'response2_correct': r2c,
        'response1_preferred': r1p, 'preference_correct': preference_correct,
        'preference_logic': preference_logic, 'reasoning': reasoning,
        'ground_truth_full_response': ground_truth_full_response
    }
    if correct_response:
        result['gpt_generated_correct_response'] = correct_response
    return result


def _make_llama_item(question: str, chosen: str, rejected: str) -> Dict:
    return {
        "conversations": [{"from": "human", "value": question}],
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected}
    }


def process_dataset(client_config: Dict[str, str], deployment_name: str,
                    dataset: List[Dict[str, Any]], output_eval_file: str,
                    output_train_file: str, dataset_type: str = "gsm8k") -> Dict[str, Any]:
    results = []
    stats = {
        'total_processed': 0, 'preference_correct': 0, 'preference_incorrect': 0,
        'preference_unknown': 0, 'errors': 0, 'correct_vs_incorrect': 0,
        'incorrect_vs_correct': 0, 'both_correct': 0, 'both_incorrect': 0,
        'both_incorrect_gpt_corrected': 0, 'both_incorrect_ground_truth_corrected': 0,
        'unknown_logic': 0
    }

    print(f"Processing {len(dataset)} pairs...")

    for i, pair in enumerate(dataset):
        if (i + 1) % 25 == 0 or i + 1 == len(dataset):
            print(f"  {i + 1}/{len(dataset)}")

        try:
            question = extract_question_from_prompt(pair['prompt'])
            positive_response = pair['positive_response']
            negative_response = pair['negative_response']
            gt_full = pair.get('ground_truth_full_response', None)

            evaluation = evaluate_preference_pair(
                client_config, deployment_name, question,
                positive_response, negative_response, gt_full,
            )

            if 'error' in evaluation:
                stats['errors'] += 1
                continue

            correct_response = None
            if (evaluation.get('response1_correct') is False and
                    evaluation.get('response2_correct') is False):
                if gt_full:
                    correct_response = gt_full
                else:
                    correct_response = generate_correct_response(client_config, deployment_name, question)
                    time.sleep(1)

            preference_analysis = determine_preference_correctness(
                evaluation, positive_response, negative_response, gt_full, correct_response
            )

            result = {
                'original_pair': pair, 'question': question,
                'positive_response': positive_response, 'negative_response': negative_response,
                'evaluation': evaluation, 'preference_analysis': preference_analysis
            }
            results.append(result)
            stats['total_processed'] += 1

            if preference_analysis['preference_correct'] is True:
                stats['preference_correct'] += 1
            elif preference_analysis['preference_correct'] is False:
                stats['preference_incorrect'] += 1
            else:
                stats['preference_unknown'] += 1

            logic = preference_analysis.get('preference_logic', 'unknown_logic')
            if logic in stats:
                stats[logic] += 1
            else:
                stats['unknown_logic'] += 1

            time.sleep(1)

        except Exception as e:
            print(f"  Error processing pair {i + 1}: {e}")
            stats['errors'] += 1
            continue

    with open(output_eval_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation to {output_eval_file}")

    # Generate LLaMA-Factory training format
    llama_data = []
    for r in results:
        r1c = r['preference_analysis']['response1_correct']
        r2c = r['preference_analysis']['response2_correct']
        q = r['question']

        if r1c is True and r2c is False:
            llama_data.append(_make_llama_item(q, r['positive_response'], r['negative_response']))
        elif r1c is True and r2c is True:
            if r['preference_analysis']['preference_correct'] is True:
                llama_data.append(_make_llama_item(q, r['positive_response'], r['negative_response']))
            else:
                llama_data.append(_make_llama_item(q, r['negative_response'], r['positive_response']))
        elif r1c is False and r2c is False:
            if 'gpt_generated_correct_response' in r['preference_analysis']:
                chosen = r['preference_analysis'].get('ground_truth_full_response') or \
                         r['preference_analysis']['gpt_generated_correct_response']
                llama_data.append(_make_llama_item(q, chosen, r['positive_response']))
        elif r1c is False and r2c is True:
            llama_data.append(_make_llama_item(q, r['negative_response'], r['positive_response']))

    with open(output_train_file, 'w') as f:
        json.dump(llama_data, f, indent=2)
    print(f"Saved {len(llama_data)} training pairs to {output_train_file}")

    return stats


def _print_stats(name: str, stats: Dict[str, Any]) -> None:
    processed = stats['total_processed']
    accuracy = (stats['preference_correct'] / processed) if processed else 0.0
    print(
        f"[{name}] processed={processed} correct={stats['preference_correct']} "
        f"incorrect={stats['preference_incorrect']} unknown={stats['preference_unknown']} "
        f"errors={stats['errors']} accuracy={accuracy:.3f}"
    )
    print(
        f"  logic: c/i={stats['correct_vs_incorrect']} i/c={stats['incorrect_vs_correct']} "
        f"both_c={stats['both_correct']} both_i={stats['both_incorrect']} "
        f"both_i_gt={stats.get('both_incorrect_ground_truth_corrected', 0)} "
        f"both_i_gpt={stats.get('both_incorrect_gpt_corrected', 0)} "
        f"unknown={stats['unknown_logic']}"
    )


def main():
    parser = argparse.ArgumentParser(description='Oracle evaluation using GPT-5')
    parser.add_argument('--dataset-type', type=str, default='gsm8k',
                        choices=['gsm8k', 'math', 'webinstruct'])
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--azure_api_key', type=str, required=True)
    parser.add_argument('--azure_endpoint', type=str, required=True)
    parser.add_argument('--azure_api_version', type=str, default="2025-01-01-preview")
    parser.add_argument('--deployment_name', type=str, default="gpt-5-chat")
    parser.add_argument('--high_consistent_file', type=str, required=True)
    parser.add_argument('--oracle_file', type=str, required=True)
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    set_dataset_type(args.dataset_type)
    paths = get_paths(args.output_root, args.dataset, args.iteration)
    paths.makedirs()

    client_config = setup_azure_client(args.azure_api_key, args.azure_endpoint, args.azure_api_version)

    with open(args.high_consistent_file, 'r') as f:
        high_consistent_data = json.load(f)
    with open(args.oracle_file, 'r') as f:
        oracle_data = json.load(f)

    if args.max_pairs:
        random.seed(args.random_seed)
        high_consistent_data = random.sample(high_consistent_data, min(args.max_pairs, len(high_consistent_data)))
        oracle_data = random.sample(oracle_data, min(args.max_pairs, len(oracle_data)))

    print(f"Loaded {len(high_consistent_data)} high-consistency pairs and {len(oracle_data)} oracle pairs")

    high_stats = process_dataset(
        client_config, args.deployment_name, high_consistent_data,
        paths.eval_high_conf(), paths.train_high_conf(), args.dataset_type,
    )
    oracle_stats = process_dataset(
        client_config, args.deployment_name, oracle_data,
        paths.eval_oracle(), paths.train_oracle(), args.dataset_type,
    )

    _print_stats('high_consistent', high_stats)
    _print_stats('oracle', oracle_stats)

    # Probe training labels from high-consistency data
    try:
        with open(paths.eval_high_conf(), 'r') as f:
            hc_results = json.load(f)

        probe_labels = []
        for i, result in enumerate(hc_results):
            label = {
                'pair_id': i,
                'question': result['question'],
                'consistency_score': result['original_pair'].get('consistency_score'),
                'most_voted_answer': result['original_pair'].get('most_voted_answer'),
                'least_voted_answer': result['original_pair'].get('least_voted_answer'),
                'ground_truth': result['original_pair'].get('ground_truth'),
                'ground_truth_full_response': result['original_pair'].get('ground_truth_full_response'),
                'original_accuracy': result['original_pair'].get('accuracy'),
                'preference_correct': result['preference_analysis']['preference_correct'],
                'response1_correct': result['preference_analysis']['response1_correct'],
                'response2_correct': result['preference_analysis']['response2_correct'],
                'response1_preferred': result['preference_analysis']['response1_preferred'],
                'preference_logic': result['preference_analysis']['preference_logic'],
                'reasoning': result['preference_analysis']['reasoning']
            }
            if 'gpt_generated_correct_response' in result['preference_analysis']:
                label['gpt_generated_correct_response'] = result['preference_analysis']['gpt_generated_correct_response']
            probe_labels.append(label)

        with open(paths.probe_labels(), 'w') as f:
            json.dump(probe_labels, f, indent=2)
        print(f"Saved {len(probe_labels)} probe labels to {paths.probe_labels()}")

        correct_count = sum(1 for l in probe_labels if l['preference_correct'] is True)
        incorrect_count = sum(1 for l in probe_labels if l['preference_correct'] is False)
        unknown_count = sum(1 for l in probe_labels if l['preference_correct'] is None)
        gpt_corrected = sum(1 for l in probe_labels if 'gpt_generated_correct_response' in l)

        probe_summary = {
            'total_labels': len(probe_labels),
            'preference_correct': correct_count,
            'preference_incorrect': incorrect_count,
            'preference_unknown': unknown_count,
            'gpt_corrected_count': gpt_corrected,
            'accuracy': correct_count / len(probe_labels) if probe_labels else 0
        }
        with open(paths.probe_summary(), 'w') as f:
            json.dump(probe_summary, f, indent=2)

    except Exception as e:
        print(f"Error creating probe training labels: {e}")


if __name__ == "__main__":
    main()

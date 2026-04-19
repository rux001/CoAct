#!/usr/bin/env python3

import json
import argparse
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from paths import get_paths
from utils import get_task_instruction


def load_data(oracle_eval_path, high_conf_eval_path, icl_high_conf_path=None, self_label_path=None):
    with open(oracle_eval_path, "r") as f:
        oracle_data = json.load(f)
    with open(high_conf_eval_path, "r") as f:
        high_conf_data = json.load(f)

    print(f"Loaded {len(oracle_data)} oracle pairs and {len(high_conf_data)} high-consistent pairs")

    icl_data = None
    if icl_high_conf_path and Path(icl_high_conf_path).exists():
        with open(icl_high_conf_path, "r") as f:
            icl_data = json.load(f)
        print(f"Loaded {len(icl_data)} ICL high-consistency pairs")

    self_label_data = None
    if self_label_path and Path(self_label_path).exists():
        with open(self_label_path, "r") as f:
            self_label_data = json.load(f)
        print(f"Loaded {len(self_label_data)} self-label pairs")

    return oracle_data, high_conf_data, icl_data, self_label_data


def create_preference_pairs(oracle_data, high_conf_data, icl_data=None, self_label_data=None,
                            dataset_prefix='gsm8k'):
    task_instruction = get_task_instruction(dataset_prefix)
    preference_pairs: List[Dict] = []

    def _build_pair(human_value, chosen, rejected, source):
        return {
            'conversations': [{'from': 'human', 'value': human_value}],
            'chosen': {'from': 'gpt', 'value': chosen},
            'rejected': {'from': 'gpt', 'value': rejected},
            'source': source,
        }

    def _resolve_oracle_pair(item):
        evaluation = item.get('evaluation', {})
        r1_correct = evaluation.get('response1_correct', False)
        r2_correct = evaluation.get('response2_correct', False)
        r1_preferred = evaluation.get('response1_preferred', False)

        question = item.get('question', '')
        pos = item.get('positive_response', '')
        neg = item.get('negative_response', '')
        if not question or not pos or not neg:
            return None

        if r1_correct and not r2_correct:
            return question, pos, neg
        if r2_correct and not r1_correct:
            return question, neg, pos
        if r1_preferred:
            return question, pos, neg
        return question, neg, pos

    for source, data in (('oracle_eval', oracle_data), ('high_conf_eval', high_conf_data)):
        for item in data:
            resolved = _resolve_oracle_pair(item)
            if resolved is None:
                continue
            question, chosen, rejected = resolved
            preference_pairs.append(_build_pair(task_instruction + "Question: " + question, chosen, rejected, source))

    if self_label_data:
        for item in self_label_data:
            prompt = item.get('prompt', '')
            pos = item.get('positive_response', '')
            neg = item.get('negative_response', '')
            if not prompt or not pos or not neg:
                continue
            if '### Instruction:' in prompt and 'Question:' in prompt:
                question_part = prompt.split('Question:')[1].split('### Response:')[0].strip()
                human_value = task_instruction + "Question: " + question_part
            else:
                human_value = task_instruction + "Question: " + prompt
            preference_pairs.append(_build_pair(human_value, pos, neg, 'self_label'))

    if icl_data:
        for item in icl_data:
            if 'conversations' in item and 'chosen' in item and 'rejected' in item:
                preference_pairs.append({
                    'conversations': item['conversations'],
                    'chosen': item['chosen'],
                    'rejected': item['rejected'],
                    'source': 'icl_high_conf',
                })

    source_counts = {src: sum(1 for p in preference_pairs if p['source'] == src)
                     for src in ('oracle_eval', 'high_conf_eval', 'self_label', 'icl_high_conf')}
    print(
        f"Built {len(preference_pairs)} preference pairs: "
        + " ".join(f"{k}={v}" for k, v in source_counts.items())
    )
    return preference_pairs


def save_preference_dataset(preference_pairs, output_path):
    llamafactory_format = [
        {'conversations': p['conversations'], 'chosen': p['chosen'], 'rejected': p['rejected']}
        for p in preference_pairs
    ]
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(llamafactory_format, f, indent=2)
    print(f"Saved {len(llamafactory_format)} preference pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Create preference dataset from oracle data')
    parser.add_argument('--oracle_eval', type=str, required=True)
    parser.add_argument('--high_conf_eval', type=str, required=True)
    parser.add_argument('--self_label', type=str, default=None)
    parser.add_argument('--icl_high_conf', type=str, default=None)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['gsm8k', 'math', 'webinstruct'])
    parser.add_argument('--iteration', type=int, required=True)

    args = parser.parse_args()

    paths = get_paths(args.output_root, args.dataset, args.iteration)
    paths.makedirs()

    oracle_data, high_conf_data, icl_data, self_label_data = load_data(
        args.oracle_eval, args.high_conf_eval, args.icl_high_conf, args.self_label,
    )
    preference_pairs = create_preference_pairs(
        oracle_data, high_conf_data, icl_data, self_label_data, dataset_prefix=args.dataset,
    )
    save_preference_dataset(preference_pairs, paths.preference_pairs())


if __name__ == "__main__":
    main()

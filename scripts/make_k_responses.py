#!/usr/bin/env python3
"""Aggregate per-temperature JSONL runs into a single K-responses JSON file."""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List


def read_jsonl(file_path: str) -> List[dict]:
    records: List[dict] = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_k_responses(input_dir: str, output: str) -> str:
    pattern = os.path.join(input_dir, "run*_t*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL files found with pattern: {pattern}")

    prompt_to_responses: Dict[str, List[dict]] = defaultdict(list)

    for file_path in files:
        records = read_jsonl(file_path)
        temperature = None
        base_name = os.path.basename(file_path)
        if "_t" in base_name and base_name.endswith(".jsonl"):
            try:
                temp_str = base_name.split("_t", 1)[1].rsplit(".jsonl", 1)[0]
                temperature = float(temp_str)
            except Exception:
                temperature = None

        for rec in records:
            prompt = rec.get("prompt")
            response_text = rec.get("predict")
            if not prompt or response_text is None:
                continue
            prompt_to_responses[prompt].append({
                "response_index": len(prompt_to_responses[prompt]),
                "temperature": temperature,
                "response": response_text,
                "is_empty": False
            })

    k_responses_records: List[dict] = []
    for prompt, responses in prompt_to_responses.items():
        k_responses_records.append({
            "original_sample": {
                "instruction": prompt,
                "input": "",
                "output": ""
            },
            "responses": responses
        })

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(k_responses_records, f, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    out = build_k_responses(args.input_dir, args.output)
    print(f"Wrote K-responses -> {out}")


if __name__ == "__main__":
    main()

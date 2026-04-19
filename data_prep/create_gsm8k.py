#!/usr/bin/env python3
"""GSM8K dataset creator for LLaMA-Factory.

Writes train/dev/test splits (6.7K / 0.8K / 1.3K) and four active-learning
iteration splits (it0..it3, 1.675K each) to `${LLAMAFACTORY_DIR}/data/`.
"""

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are given a grade school math word problem involving basic arithmetic, "
    "algebra, or geometry. Your task is to carefully read the problem and provide "
    "a step-by-step solution for it.\nProvide a step-by-step reasoning process and "
    "then write the final numerical answer on a new line in the format: "
    "final answer: <answer>."
)

_FINAL_ANSWER_PATTERNS = [
    r"final answer:\s*<?([0-9,.-]+)>?",
    r"####\s*([0-9,.-]+)",
    r"(?:the\s+)?answer\s*:?\s*is\s*([0-9,.-]+)",
]


def _resolve_data_dir() -> str:
    base = os.environ.get("LLAMAFACTORY_DATA_DIR") or os.path.join(
        os.environ.get("LLAMAFACTORY_DIR", ""), "data"
    )
    if not base or base == "data":
        raise EnvironmentError(
            "Set LLAMAFACTORY_DIR (or LLAMAFACTORY_DATA_DIR) to your "
            "LLaMA-Factory data directory."
        )
    os.makedirs(base, exist_ok=True)
    return base


def extract_final_answer(answer_text: str) -> str:
    for pattern in _FINAL_ANSWER_PATTERNS:
        match = re.search(pattern, answer_text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").strip()

    for line in reversed(answer_text.strip().split("\n")):
        line = line.strip()
        if re.match(r"^[0-9,.-]+$", line):
            return line.replace(",", "").strip()

    numbers = re.findall(r"[0-9,.-]+", answer_text)
    return numbers[-1].replace(",", "").strip() if numbers else ""


def convert_to_instruction_format(sample: Dict[str, Any]) -> Dict[str, Any] | None:
    question = sample.get("question", "")
    answer = sample.get("answer", "")
    if not question or not answer:
        return None

    final_answer = extract_final_answer(answer)
    if not final_answer:
        return None

    return {
        "instruction": f"{SYSTEM_INSTRUCTION}\n\nQuestion: {question}",
        "input": "",
        "output": f"{answer.strip()}\n\nfinal answer: {final_answer}",
    }


def create_instruction_dataset(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = [c for c in (convert_to_instruction_format(s) for s in samples) if c]
    skipped = len(samples) - len(converted)
    logger.info("Converted %d samples (%d skipped)", len(converted), skipped)
    return converted


def split_train_dev_test(
    train_data: List[Dict], train_size: int = 6700, dev_size: int = 800, test_size: int = 1300
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(42)
    data = list(train_data)
    random.shuffle(data)

    need = train_size + dev_size + test_size
    if len(data) < need:
        ratio = len(data) / need
        train_size = int(train_size * ratio)
        dev_size = int(dev_size * ratio)
        test_size = len(data) - train_size - dev_size

    train = data[:train_size]
    dev = data[train_size:train_size + dev_size]
    test = data[train_size + dev_size:train_size + dev_size + test_size]
    logger.info("Splits: train=%d dev=%d test=%d", len(train), len(dev), len(test))
    return train, dev, test


def split_al_iterations(
    train_data: List[Dict], split_size: int = 1675, num_iters: int = 4
) -> List[List[Dict]]:
    random.seed(42)
    data = list(train_data)
    random.shuffle(data)

    need = split_size * num_iters
    if len(data) < need:
        split_size = int(split_size * (len(data) / need))

    splits = [data[i * split_size:(i + 1) * split_size] for i in range(num_iters)]
    logger.info("AL splits: %s", "/".join(str(len(s)) for s in splits))
    return splits


def save_json(samples: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d samples -> %s", len(samples), path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = _resolve_data_dir()
    ds = load_dataset("gsm8k", "main")
    if "train" not in ds:
        raise RuntimeError("GSM8K train split not available")

    train_raw, dev_raw, test_raw = split_train_dev_test(list(ds["train"]))
    save_json(create_instruction_dataset(train_raw), f"{data_dir}/gsm8k_AL_train.json")
    save_json(create_instruction_dataset(dev_raw), f"{data_dir}/gsm8k_AL_dev.json")
    save_json(create_instruction_dataset(test_raw), f"{data_dir}/gsm8k_AL_test.json")

    for i, split in enumerate(split_al_iterations(list(ds["train"]))):
        save_json(create_instruction_dataset(split), f"{data_dir}/gsm8k_al_it{i}.json")


if __name__ == "__main__":
    main()

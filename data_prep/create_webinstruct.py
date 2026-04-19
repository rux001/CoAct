#!/usr/bin/env python3
"""WebInstruct-verified dataset creator for LLaMA-Factory.

Writes four active-learning iteration splits (it0..it3, 2K each, filtered to
`answer_type='Float'`) and a test split (Physics, Float) to
`${LLAMAFACTORY_DIR}/data/`.
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are given a scientific problem that requires mathematical and "
    "physical reasoning and calculations. Your task is to carefully read the "
    "problem and provide a step-by-step solution for it.\n"
    "Provide a step-by-step reasoning process and then write the final "
    "numerical answer on a new line in the format: final answer: <answer>."
)


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


def extract_answer_and_reasoning(sample: Dict[str, Any]) -> Tuple[str, str]:
    reasoning = sample.get("solution", sample.get("explanation", "")) or ""
    final_answer = sample.get("answer", "") or ""
    return str(reasoning).strip(), str(final_answer).strip()


def convert_to_instruction_format(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = sample.get("question", sample.get("problem", sample.get("text", "")))
    reasoning, final_answer = extract_answer_and_reasoning(sample)
    if not question or not final_answer:
        return None

    output = (
        f"{reasoning}\n\nfinal answer: {final_answer}"
        if reasoning
        else f"final answer: {final_answer}"
    )
    return {
        "instruction": f"{SYSTEM_INSTRUCTION}\n\nQuestion: {question}",
        "input": "",
        "output": output,
    }


def create_instruction_dataset(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = [c for c in (convert_to_instruction_format(s) for s in samples) if c]
    logger.info("Converted %d samples (%d skipped)", len(converted), len(samples) - len(converted))
    return converted


def filter_float_samples(
    dataset, target_count: Optional[int] = None, category: Optional[str] = None
) -> List[Dict]:
    filtered: List[Dict] = []
    for sample in dataset:
        if sample.get("answer_type", "") != "Float":
            continue
        if category and sample.get("category", sample.get("subject", "")) != category:
            continue
        filtered.append(sample)

    if target_count and len(filtered) > target_count:
        random.seed(42)
        filtered = random.sample(filtered, target_count)

    logger.info(
        "Filtered %d Float samples%s",
        len(filtered),
        f" (category={category})" if category else "",
    )
    return filtered


def split_al_iterations(
    data: List[Dict], split_size: int = 2000, num_iters: int = 4
) -> List[List[Dict]]:
    random.seed(42)
    shuffled = list(data)
    random.shuffle(shuffled)
    need = split_size * num_iters
    if len(shuffled) < need:
        split_size = int(split_size * (len(shuffled) / need))
    return [shuffled[i * split_size:(i + 1) * split_size] for i in range(num_iters)]


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
    ds = load_dataset("TIGER-Lab/WebInstruct-verified")

    if "train" not in ds:
        raise RuntimeError("WebInstruct-verified train split not available")

    train_float = filter_float_samples(ds["train"], target_count=8000)
    for i, split in enumerate(split_al_iterations(train_float)):
        save_json(create_instruction_dataset(split), f"{data_dir}/webinstruct_al_it{i}.json")

    if "test" in ds:
        test_float = filter_float_samples(ds["test"], category="Physics")
        if test_float:
            save_json(
                create_instruction_dataset(test_float),
                f"{data_dir}/webinstruct_AL_test.json",
            )
    else:
        logger.warning("Test split not found; skipping test set creation")


if __name__ == "__main__":
    main()

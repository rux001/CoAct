#!/usr/bin/env python3
"""MATH dataset creator for LLaMA-Factory.

Writes train/dev/test splits (6.7K train, 0.8K dev, all upstream test) and
four active-learning iteration splits (it0..it3, 1.675K each) to
`${LLAMAFACTORY_DIR}/data/`. Expects `MATH.zip` alongside this script.
"""

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Tuple

import datasets

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a competition-level mathematics expert. Provide a rigorous, "
    "step-by-step solution to each problem, clearly justifying every "
    "transformation. Present the final result inside a LaTeX boxed expression, "
    "i.e., write the answer as \\boxed{<answer>} with no additional text."
)

_CITATION = """\
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora
    and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={arXiv preprint arXiv:2103.03874},
  year={2021}
}
"""
_DESCRIPTION = (
    "The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of "
    "problems from mathematics competitions. This script processes the local "
    "zip file."
)
_URL = "MATH.zip"


class CompetitionMathDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "problem": datasets.Value("string"),
                "level": datasets.Value("string"),
                "type": datasets.Value("string"),
                "solution": datasets.Value("string"),
            }),
            homepage="https://github.com/hendrycks/math",
            license="https://github.com/hendrycks/math/blob/main/LICENSE",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        download_dir = dl_manager.download_and_extract(_URL)
        math_path = os.path.join(download_dir, "MATH")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": dl_manager.iter_files(os.path.join(math_path, "train"))},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": dl_manager.iter_files(os.path.join(math_path, "test"))},
            ),
        ]

    def _generate_examples(self, data_dir):
        for idx, filepath in enumerate(data_dir):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    content = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON file: %s", filepath)
                    continue
                yield idx, {
                    "problem": content.get("problem", ""),
                    "level": content.get("level", ""),
                    "type": content.get("type", ""),
                    "solution": content.get("solution", ""),
                }


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


def extract_boxed_answer(solution_text: str) -> str:
    for pattern in (
        r"\$\\boxed\{([^}]+)\}\$",
        r"\\boxed\{([^}]+)\}",
        r"\$\$\\boxed\{([^}]+)\}\$\$",
    ):
        match = re.search(pattern, solution_text)
        if match:
            return match.group(1).strip()

    for pattern in (
        r"(?:the\s+)?answer\s*:?\s*is\s*([^\n]+)",
        r"final\s+answer\s*:?\s*([^\n]+)",
        r"solution\s*:?\s*([^\n]+)",
    ):
        match = re.search(pattern, solution_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    for line in reversed(solution_text.strip().split("\n")):
        line = line.strip()
        if line and not line.startswith("\\") and len(line) < 100:
            return line
    return ""


def convert_to_instruction_format(sample: Dict[str, Any]) -> Dict[str, Any] | None:
    problem = sample.get("problem", "")
    solution = sample.get("solution", "")
    if not problem or not solution:
        return None

    final_answer = extract_boxed_answer(solution)
    if not final_answer:
        return None

    if not solution.strip().endswith("$") and "\\boxed{" not in solution:
        formatted = f"{solution.strip()}\n\nTherefore, the answer is $\\boxed{{{final_answer}}}$"
    else:
        formatted = solution.strip()

    return {"instruction": SYSTEM_INSTRUCTION, "input": problem, "output": formatted}


def create_instruction_dataset(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = [c for c in (convert_to_instruction_format(s) for s in samples) if c]
    logger.info("Converted %d samples (%d skipped)", len(converted), len(samples) - len(converted))
    return converted


def split_train_dev(
    train_data: List[Dict], train_size: int = 6700, dev_size: int = 800
) -> Tuple[List[Dict], List[Dict]]:
    random.seed(42)
    data = list(train_data)
    random.shuffle(data)
    need = train_size + dev_size
    if len(data) < need:
        ratio = len(data) / need
        train_size = int(train_size * ratio)
        dev_size = len(data) - train_size
    return data[:train_size], data[train_size:train_size + dev_size]


def split_al_iterations(
    train_data: List[Dict], split_size: int = 1675, num_iters: int = 4
) -> List[List[Dict]]:
    random.seed(42)
    data = list(train_data)
    random.shuffle(data)
    need = split_size * num_iters
    if len(data) < need:
        split_size = int(split_size * (len(data) / need))
    return [data[i * split_size:(i + 1) * split_size] for i in range(num_iters)]


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
    ds = datasets.load_dataset(os.path.abspath(__file__))
    if "train" not in ds or "test" not in ds:
        raise RuntimeError("MATH dataset missing train or test split")

    train_raw, dev_raw = split_train_dev(list(ds["train"]))
    save_json(create_instruction_dataset(train_raw), f"{data_dir}/math_AL_train.json")
    save_json(create_instruction_dataset(dev_raw), f"{data_dir}/math_AL_dev.json")
    save_json(create_instruction_dataset(list(ds["test"])), f"{data_dir}/math_AL_test.json")

    for i, split in enumerate(split_al_iterations(list(ds["train"]))):
        save_json(create_instruction_dataset(split), f"{data_dir}/math_al_it{i}.json")


if __name__ == "__main__":
    main()

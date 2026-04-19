"""Centralized path construction for the Selective DPO pipeline.

All output paths are derived from (output_root, dataset, iteration).
Directory structure encodes dataset and iteration, keeping filenames short.
"""

import os
from dataclasses import dataclass


@dataclass
class IterationPaths:
    """All output paths for a single pipeline iteration."""
    base: str
    responses: str
    consistency: str
    oracle: str
    icl: str
    training: str
    model: str

    # Response files
    def response_run(self, run: int, temp: float) -> str:
        return os.path.join(self.responses, f"run{run}_t{temp:.2f}.jsonl")

    def k_responses(self) -> str:
        return os.path.join(self.responses, "k_responses.json")

    def response_glob(self) -> str:
        return os.path.join(self.responses, "run*_t*.jsonl")

    # Consistency files
    def all_pairs(self) -> str:
        return os.path.join(self.consistency, "all_pairs.json")

    def self_label(self) -> str:
        return os.path.join(self.consistency, "self_label.json")

    def oracle_candidates(self) -> str:
        return os.path.join(self.consistency, "oracle_candidates.json")

    def high_conf_candidates(self) -> str:
        return os.path.join(self.consistency, "high_conf_candidates.json")

    # Oracle files
    def eval_oracle(self) -> str:
        return os.path.join(self.oracle, "eval_oracle.json")

    def eval_high_conf(self) -> str:
        return os.path.join(self.oracle, "eval_high_conf.json")

    def train_oracle(self) -> str:
        return os.path.join(self.oracle, "train_oracle.json")

    def train_high_conf(self) -> str:
        return os.path.join(self.oracle, "train_high_conf.json")

    def probe_labels(self) -> str:
        return os.path.join(self.oracle, "probe_labels.json")

    def probe_summary(self) -> str:
        return os.path.join(self.oracle, "probe_summary.json")

    # ICL files
    def icl_prompts(self) -> str:
        return os.path.join(self.icl, "prompts.json")

    def icl_questions(self) -> str:
        return os.path.join(self.icl, "questions.jsonl")

    def icl_dataset(self) -> str:
        return os.path.join(self.icl, "dataset.json")

    def icl_run(self, run: int, temp: float) -> str:
        return os.path.join(self.icl, f"run{run}_t{temp:.2f}.jsonl")

    def icl_high_consistent(self) -> str:
        return os.path.join(self.icl, "high_consistent.json")

    def icl_all_with_scores(self) -> str:
        return os.path.join(self.icl, "all_with_scores.json")

    def icl_response_glob(self) -> str:
        return os.path.join(self.icl, "run*_t*.jsonl")

    # Training files
    def preference_pairs(self) -> str:
        return os.path.join(self.training, "preference_pairs.json")

    # Model files
    def merged_model(self) -> str:
        return os.path.join(self.model, "merged")

    def makedirs(self):
        for d in [self.responses, self.consistency, self.oracle,
                  self.icl, self.training, self.model]:
            os.makedirs(d, exist_ok=True)


def get_paths(output_root: str, dataset: str, iteration: int) -> IterationPaths:
    base = os.path.join(output_root, dataset, f"it{iteration}")
    return IterationPaths(
        base=base,
        responses=os.path.join(base, "responses"),
        consistency=os.path.join(base, "consistency"),
        oracle=os.path.join(base, "oracle"),
        icl=os.path.join(base, "icl"),
        training=os.path.join(base, "training"),
        model=os.path.join(base, "model"),
    )

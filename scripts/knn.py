#!/usr/bin/env python3
"""KNN-based selector for identifying self-consistent errors in high-consistency preference pairs.

Used by the active-learning pipeline (scripts/self_consistency.py) to select the top-M_high
most-OOD samples from D_high^(t) for oracle labeling, following Eq. for S_high^(t) in the paper.
"""

import logging
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class KNNErrorDetector:
    """KNN-based detector using oracle-verified correct instructions as ID reference."""

    def __init__(self, model, tokenizer, k: int = 50, device: str = 'cuda', max_length: int = 2048):
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.device = device
        self.max_length = max_length
        self.Z_correct: np.ndarray = np.empty((0, 0), dtype='float32')
        self.index_correct: Optional[faiss.Index] = None
        self.embedding_dim: Optional[int] = None

    def get_question_embedding(self, question: str) -> np.ndarray:
        prompt_ids = self.tokenizer(question, return_tensors='pt')['input_ids'].to(self.device)
        with torch.no_grad():
            outputs = self.model(prompt_ids, output_hidden_states=True)
            h = torch.mean(outputs.hidden_states[-1][0], dim=0)
        return h.cpu().numpy()

    def get_question_embeddings_batch(self, questions: List[str], batch_size: int = 8) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(questions), batch_size), desc="Extracting question embeddings"):
            for question in questions[i:i + batch_size]:
                try:
                    emb = self.get_question_embedding(question)
                    emb = F.normalize(torch.from_numpy(emb), p=2, dim=-1).numpy().astype('float32')
                    all_embeddings.append(emb)
                except Exception:
                    fallback_dim = all_embeddings[-1].shape[0] if all_embeddings else 4096
                    all_embeddings.append(np.zeros(fallback_dim, dtype='float32'))
        return np.array(all_embeddings, dtype='float32')

    def build_reference_sets_from_oracle_evaluation(self, oracle_data: List[Dict], batch_size: int = 8):
        """Build the ID reference set from oracle-verified correct preferences."""
        logger.info(f"Building reference sets from {len(oracle_data)} oracle samples")
        id_questions = [
            item['question'] for item in oracle_data
            if item.get('evaluation', {}).get('response1_correct')
        ]
        logger.info(f"Collected {len(id_questions)} ID questions")
        if not id_questions:
            raise ValueError("No ID questions found. Check your oracle data.")

        self.Z_correct = self.get_question_embeddings_batch(id_questions, batch_size=batch_size)
        self.embedding_dim = self.Z_correct.shape[1]
        logger.info(f"Reference: {len(self.Z_correct)} ID questions, dim={self.embedding_dim}")
        self._build_indices()

    def _build_indices(self):
        if len(self.Z_correct) == 0:
            raise ValueError("Cannot build index without reference data")
        self.index_correct = faiss.IndexFlatL2(self.embedding_dim)
        self.index_correct.add(self.Z_correct)

    def knn_distance(self, z: np.ndarray, index: faiss.Index) -> float:
        """r_k(z) = L2 distance to the k-th nearest reference neighbor."""
        if index is None or index.ntotal == 0:
            return float('inf')
        k_actual = min(self.k, index.ntotal)
        distances, _ = index.search(z.reshape(1, -1), k_actual)
        return float(distances[0, -1])

    def compute_knn_scores_batch(self, questions: List[str], responses: List[str] = None,
                                 batch_size: int = 8) -> List[Dict[str, float]]:
        embeddings = self.get_question_embeddings_batch(questions, batch_size)
        scores = []
        for z in embeddings:
            r_correct = self.knn_distance(z, self.index_correct)
            scores.append({'ood_score': r_correct, 'id_score': -r_correct, 'r_correct': r_correct})
        return scores

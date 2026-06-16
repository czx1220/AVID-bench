#!/usr/bin/env python3
"""
Compute Segment-level Sentence Cosine Similarity metric.
Only evaluates negative samples correctly identified as inconsistent
(exists_correct=True and ground_truth.exists="Yes").
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    sys.exit(1)


def get_all_result_files(results_dir: str) -> List[Path]:
    """Get all *_result.json files from the directory."""
    results_path = Path(results_dir)
    return sorted([f for f in results_path.glob('*_result.json')])


def filter_correct_neg_samples(result_files: List[Path]) -> List[Dict[str, Any]]:
    """Filter for negative samples correctly identified as inconsistent."""
    samples = []
    for filepath in result_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip error samples
            if data.get('error'):
                continue

            gt = data.get('ground_truth', {})
            # Only evaluate correctly identified inconsistent samples
            if data.get('exists_correct') is True and gt.get('exists') == 'Yes':
                gt_text = gt.get('inconsistency_point', '')
                pred_text = data.get('parsed_answer', {}).get('inconsistency_point', '')
                if gt_text and pred_text:
                    samples.append({
                        'qa_id': data.get('qa_id', filepath.stem),
                        'gt_text': gt_text,
                        'pred_text': pred_text,
                    })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return samples


def compute_sentence_cosine(samples: List[Dict[str, Any]], model: SentenceTransformer) -> Dict[str, Any]:
    """Batch compute sentence cosine similarity."""
    if not samples:
        return {'avg_sentence_cosine': 0.0, 'num_samples': 0, 'scores': []}

    gt_texts = [s['gt_text'] for s in samples]
    pred_texts = [s['pred_text'] for s in samples]

    # Batch encode
    gt_embeddings = model.encode(gt_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    pred_embeddings = model.encode(pred_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # Pairwise cosine similarity (already normalized, so dot product = cosine)
    scores = np.sum(gt_embeddings * pred_embeddings, axis=1).tolist()

    return {
        'avg_sentence_cosine': float(np.mean(scores)),
        'num_samples': len(scores),
        'scores': scores,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute Segment-level Sentence Cosine Similarity')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to directory containing *_result.json files')
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Sentence-transformers model name or path')
    args = parser.parse_args()

    print(f"Loading model: {args.model_name} ...")
    model = SentenceTransformer(args.model_name)
    print(f"Model loaded.\n")

    result_dir = Path(args.results_dir)
    result_files = get_all_result_files(str(result_dir))
    if not result_files:
        print(f"No *_result.json files found in {result_dir}")
        sys.exit(1)

    samples = filter_correct_neg_samples(result_files)
    stats = compute_sentence_cosine(samples, model)

    dir_name = result_dir.name
    std = float(np.std(stats['scores'])) if stats['scores'] else 0.0
    print(f"{dir_name}:")
    print(f"  Correctly identified neg samples: {stats['num_samples']}")
    print(f"  Avg Sentence Cosine Similarity: {stats['avg_sentence_cosine']:.4f} +/- {std:.4f}")


if __name__ == "__main__":
    main()

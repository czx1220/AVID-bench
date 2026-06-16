#!/usr/bin/env python3
"""
Compute Full-Video Sentence Cosine Similarity metric.
Only evaluates negative samples correctly identified as inconsistent.
For each prediction, matches to the GT interval with highest IoU, then computes
sentence-level cosine similarity between predicted and GT reasoning text.
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
    results_path = Path(results_dir)
    return sorted([f for f in results_path.glob('*_result.json')])


def parse_time_range(time_str: str):
    """Parse a time range string like 'from 1.5s to 3.0s'."""
    import re
    match = re.match(r'from\s+([\d.]+)s?\s+to\s+([\d.]+)s?', time_str, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def calculate_iou(span1: tuple, span2: tuple) -> float:
    """Compute IoU between two time spans."""
    inter_start = max(span1[0], span2[0])
    inter_end = min(span1[1], span2[1])
    intersection = max(0, inter_end - inter_start)
    union = (span1[1] - span1[0]) + (span2[1] - span2[0]) - intersection
    return intersection / union if union > 0 else 0


def collect_reasoning_pairs(result_files: List[Path]) -> List[Dict[str, str]]:
    """
    Collect text pairs for sentence cosine similarity computation.
    Logic:
    - Only neg files where exists_correct=True
    - pred_metrics are ordered by GT; each GT is matched to the pred with highest IoU
    - For each GT, find the best-matching pred reasoning
    """
    pairs = []
    for filepath in result_files:
        try:
            data = json.loads(filepath.read_text())
            is_neg = filepath.name.startswith('neg_')

            if not is_neg:
                continue
            if data.get('exists_correct') is not True:
                continue

            gt_inconsistencies = data.get('ground_truth', {}).get('inconsistencies', [])
            parsed = data.get('parsed_answer', {})
            pred_inconsistencies = parsed.get('inconsistencies', []) or parsed.get('inconsistency_points', [])
            pred_metrics = data.get('pred_metrics', [])

            # Parse GT spans and reasoning text
            gt_spans = []
            gt_points = []
            for inc in gt_inconsistencies:
                s, e = parse_time_range(inc.get('time_range', ''))
                if s is not None:
                    gt_spans.append((s, e))
                    gt_points.append(inc.get('inconsistency_point', ''))

            # For each GT, find the best-matching pred reasoning by IoU
            best_pred_for_gt = [None] * len(gt_spans)
            best_iou_for_gt = [0.0] * len(gt_spans)

            for pred_inc in pred_inconsistencies:
                pred_time_range = pred_inc.get('time_range', '')
                pred_reasoning = pred_inc.get('reasoning', '') or pred_inc.get('inconsistency_point', '')
                s, e = parse_time_range(pred_time_range)
                # Also support start_time/end_time format
                if s is None:
                    try:
                        s = float(pred_inc.get('start_time', ''))
                        e = float(pred_inc.get('end_time', ''))
                    except (ValueError, TypeError):
                        s, e = None, None

                if s is None or not pred_reasoning:
                    continue

                pred_span = (s, e)
                # Find GT with highest IoU
                best_idx = -1
                best_iou = 0
                for j, gt_span in enumerate(gt_spans):
                    iou = calculate_iou(pred_span, gt_span)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

                if best_idx >= 0 and best_iou > best_iou_for_gt[best_idx]:
                    best_iou_for_gt[best_idx] = best_iou
                    best_pred_for_gt[best_idx] = pred_reasoning

            # Collect text pairs for each GT:
            # - If a matching pred exists: compute actual sentence cosine
            # - If no pred matched (missed): count as score=0
            for i in range(len(gt_spans)):
                if i < len(pred_metrics) and pred_metrics[i].get('rouge_l') is not None:
                    gt_text = gt_points[i]
                    pred_text = best_pred_for_gt[i]
                    if gt_text and pred_text:
                        pairs.append({
                            'qa_id': data.get('qa_id', filepath.stem),
                            'gt_text': gt_text,
                            'pred_text': pred_text,
                        })
                    else:
                        # No matching pred (missed), mark as score=0
                        pairs.append({
                            'qa_id': data.get('qa_id', filepath.stem),
                            'gt_text': gt_points[i] if i < len(gt_points) else '',
                            'pred_text': None,  # Indicates no match
                        })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return pairs


def compute_sentence_cosine(pairs: List[Dict[str, str]], model: SentenceTransformer) -> Dict[str, Any]:
    if not pairs:
        return {'avg_sentence_cosine': 0.0, 'num_pairs': 0, 'std': 0.0, 'scores': [], 'valid_pairs': 0, 'null_pairs': 0}

    # Separate matched and unmatched pairs
    valid_pairs = [p for p in pairs if p['pred_text'] is not None]
    null_count = len(pairs) - len(valid_pairs)

    scores = []
    if valid_pairs:
        gt_texts = [p['gt_text'] for p in valid_pairs]
        pred_texts = [p['pred_text'] for p in valid_pairs]

        gt_emb = model.encode(gt_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        pred_emb = model.encode(pred_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

        scores = np.sum(gt_emb * pred_emb, axis=1).tolist()

    # Unmatched pairs count as 0 (consistent with missed-detection penalty)
    scores.extend([0.0] * null_count)

    return {
        'avg_sentence_cosine': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'num_pairs': len(scores),
        'valid_pairs': len(valid_pairs),
        'null_pairs': null_count,
        'scores': scores,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute Full-Video Sentence Cosine Similarity')
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

    pairs = collect_reasoning_pairs(result_files)
    stats = compute_sentence_cosine(pairs, model)

    dir_name = result_dir.name
    print(f"{dir_name}:")
    print(f"  Total pairs: {stats['num_pairs']} (matched: {stats['valid_pairs']}, unmatched: {stats['null_pairs']})")
    print(f"  Avg Sentence Cosine Similarity: {stats['avg_sentence_cosine']:.4f} +/- {stats['std']:.4f}")


if __name__ == "__main__":
    main()

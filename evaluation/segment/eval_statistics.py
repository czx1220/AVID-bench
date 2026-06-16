#!/usr/bin/env python3
"""
Compute statistics from result.json files under a results directory.
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def load_result_json(filepath: Path) -> Dict[str, Any]:
    """Load a single result.json file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_result_file(filepath: Path) -> bool:
    """Check whether the file is a result.json file (exclude summary.json)."""
    return filepath.name.endswith('_result.json')


def get_all_result_files(results_dir: str) -> List[Path]:
    """Retrieve all result.json files."""
    results_path = Path(results_dir)
    return sorted([f for f in results_path.glob('*_result.json')])


def compute_stats(result_files: List[Path]) -> Dict[str, Any]:
    """Compute evaluation statistics."""
    total = len(result_files)

    # Exists judgement statistics
    exists_correct_count = 0
    exists_incorrect_count = 0

    # Positive/negative sample statistics
    # pos_* = consistent (no inconsistency, ground_truth.exists="No")
    # neg_* = inconsistent (ground_truth.exists="Yes")
    pos_count = 0  # Positive sample count (consistent)
    pos_correct = 0  # Positive samples correctly judged
    neg_count = 0  # Negative sample count (inconsistent)
    neg_correct = 0  # Negative samples correctly judged

    # Classification accuracy and reasoning metrics: only for samples correctly judged as inconsistent
    # i.e. exists_correct=true and ground_truth.exists="Yes"
    correct_inconsistent = []  # Samples correctly judged as inconsistent

    # Reasoning metrics (only for samples correctly judged as inconsistent)
    rouge_l_scores = []
    meteor_scores = []
    bleu4_scores = []

    # Per-category injection_type statistics
    category_stats = defaultdict(lambda: {
        'total': 0,
        'type_correct': 0
    })

    error_count = 0
    error_files = []

    for filepath in result_files:
        try:
            data = load_result_json(filepath)

            # Determine file type (pos=consistent, neg=inconsistent)
            is_pos = filepath.name.startswith('pos_')
            is_neg = filepath.name.startswith('neg_')

            # Skip error samples; do not count them in statistics
            if data.get('error'):
                error_count += 1
                error_files.append(data['qa_id'])
                continue

            # Count positive/negative samples (valid only)
            if is_pos:
                pos_count += 1
            elif is_neg:
                neg_count += 1

            # Count exists judgements
            if data.get('exists_correct') is True:
                exists_correct_count += 1
                # Positive sample correctly judged
                if is_pos:
                    pos_correct += 1
                # Negative sample correctly judged
                elif is_neg:
                    neg_correct += 1
            elif data.get('exists_correct') is False:
                exists_incorrect_count += 1

            # Get ground_truth and parsed_answer
            gt = data.get('ground_truth', {})
            pa = data.get('parsed_answer', {})

            gt_exists = gt.get('exists', '')
            gt_type = gt.get('injection_type', '')

            # Only compute classification accuracy and reasoning metrics for samples
            # correctly judged as inconsistent (exists_correct=true and ground_truth.exists="Yes")
            if data.get('exists_correct') is True and gt_exists == 'Yes':
                correct_inconsistent.append(data)

                # Reasoning metrics
                if data.get('rouge_l_score') is not None:
                    rouge_l_scores.append(data['rouge_l_score'])
                if data.get('meteor_score') is not None:
                    meteor_scores.append(data['meteor_score'])
                if data.get('bleu4_score') is not None:
                    bleu4_scores.append(data['bleu4_score'])

                # Classification accuracy
                if data.get('injection_type_correct') is True:
                    category_stats[gt_type]['type_correct'] += 1
                category_stats[gt_type]['total'] += 1

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            error_count += 1
            error_files.append(filepath.name)

    # Compute accuracy (excluding error samples)
    valid_total = total - error_count  # Valid sample count (excluding errors)
    exists_accuracy = exists_correct_count / valid_total if valid_total > 0 else 0

    # Compute recall, false positive rate, etc.
    # Confusion matrix definition:
    # - TP (True Positive) = negative samples correctly judged (neg file with exists_correct=true, i.e. correctly identified inconsistency)
    # - FN (False Negative) = negative samples incorrectly judged (neg file with exists_correct=false, i.e. missed inconsistency)
    # - TN (True Negative) = positive samples correctly judged (pos file with exists_correct=true, i.e. correctly identified consistency)
    # - FP (False Positive) = positive samples incorrectly judged (pos file with exists_correct=false, i.e. false alarm of inconsistency)
    TP = neg_correct  # Correctly identified inconsistency
    FN = neg_count - neg_correct  # Missed inconsistency
    TN = pos_correct  # Correctly identified consistency
    FP = pos_count - pos_correct  # False alarm of inconsistency

    # Recall, precision, etc. are also based on valid samples (excluding errors)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall = TP / (TP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision = TP / (TP + FP)
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate = FP / (FP + TN)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # Specificity = TN / (TN + FP)

    # Overall accuracy based on valid samples
    valid_accuracy = (TP + TN) / valid_total if valid_total > 0 else 0

    # Classification accuracy (only for samples correctly judged as inconsistent)
    classification_accuracy = 0
    classification_correct = 0
    if len(correct_inconsistent) > 0:
        classification_correct = sum(1 for d in correct_inconsistent if d.get('injection_type_correct') is True)
        classification_accuracy = classification_correct / len(correct_inconsistent)

    # Average reasoning metrics (only for samples correctly judged as inconsistent)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0

    return {
        'total_samples': total,
        'valid_samples': valid_total,
        'error_count': error_count,
        'exists_correct': exists_correct_count,
        'exists_incorrect': exists_incorrect_count,
        'exists_accuracy': exists_accuracy,
        'valid_accuracy': valid_accuracy,
        # Positive/negative sample statistics
        'pos_count': pos_count,
        'pos_correct': pos_correct,
        'neg_count': neg_count,
        'neg_correct': neg_correct,
        # Confusion matrix metrics
        'TP': TP,
        'FN': FN,
        'TN': TN,
        'FP': FP,
        'recall': recall,
        'precision': precision,
        'fpr': fpr,
        'specificity': specificity,
        # Classification accuracy
        'correct_inconsistent': len(correct_inconsistent),
        'classification_correct': classification_correct,
        'classification_accuracy': classification_accuracy,
        # Reasoning metrics
        'avg_rouge_l': avg_rouge_l,
        'avg_meteor': avg_meteor,
        'avg_bleu4': avg_bleu4,
        # Per-category statistics
        'category_stats': dict(category_stats),
        'error_count': error_count,
        'error_files': error_files
    }


def print_stats(stats: Dict[str, Any]):
    """Print evaluation statistics."""
    print("=" * 60)
    print("Evaluation Statistics")
    print("=" * 60)

    print(f"\n[Overall Statistics]")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Error samples: {stats['error_count']}")
    print(f"  Valid samples: {stats['valid_samples']}")

    print(f"\n[Positive/Negative Sample Statistics]")
    print(f"  Positive (consistent, pos_*): {stats['pos_count']}, correctly judged: {stats['pos_correct']}")
    print(f"  Negative (inconsistent, neg_*): {stats['neg_count']}, correctly judged: {stats['neg_correct']}")

    print(f"\n[Confusion Matrix]")
    print(f"  TP (True Positive): {stats['TP']} - correctly identified inconsistency")
    print(f"  FN (False Negative): {stats['FN']} - missed inconsistency (judged as consistent)")
    print(f"  TN (True Negative): {stats['TN']} - correctly identified consistency")
    print(f"  FP (False Positive): {stats['FP']} - false alarm (judged as inconsistent)")

    print(f"\n[Recall, Precision, FPR]")
    print(f"  Recall = TP/(TP+FN): {stats['recall']:.2%}")
    print(f"  Precision = TP/(TP+FP): {stats['precision']:.2%}")
    print(f"  FPR = FP/(FP+TN): {stats['fpr']:.2%}")
    print(f"  Specificity = TN/(TN+FP): {stats['specificity']:.2%}")

    print(f"\n[Exists Judgement Accuracy]")
    print(f"  Correct: {stats['exists_correct']}")
    print(f"  Incorrect: {stats['exists_incorrect']}")
    print(f"  Accuracy: {stats['exists_accuracy']:.2%}")

    print(f"\n[Classification Accuracy] (only for samples correctly judged as inconsistent)")
    print(f"  Correctly judged as inconsistent (denominator): {stats['correct_inconsistent']}")
    print(f"  Classification correct (numerator): {stats['classification_correct']}")
    print(f"  Classification accuracy: {stats['classification_accuracy']:.2%}")

    if stats['category_stats']:
        print(f"\n[Per-Category Classification Accuracy]")
        for category, cat_stats in sorted(stats['category_stats'].items()):
            if cat_stats['total'] > 0:
                acc = cat_stats['type_correct'] / cat_stats['total']
                print(f"  {category}: {cat_stats['type_correct']}/{cat_stats['total']} = {acc:.2%}")

    print(f"\n[Reasoning Metrics] (only for samples correctly judged as inconsistent, average, n={stats['correct_inconsistent']})")
    print(f"  ROUGE-L: {stats['avg_rouge_l']:.2f}")
    print(f"  METEOR: {stats['avg_meteor']:.2f}")
    print(f"  BLEU-4: {stats['avg_bleu4']:.2f}")

    if stats['error_files']:
        print(f"\n[Error Files]")
        for f in stats['error_files'][:10]:
            print(f"  {f}")
        if len(stats['error_files']) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation statistics from result.json files.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the directory containing result.json files.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save the statistics report. If not specified, only prints to stdout.")
    args = parser.parse_args()

    result_files = get_all_result_files(args.results_dir)
    print(f"Results directory: {args.results_dir}")
    print(f"Found {len(result_files)} result.json files")

    stats = compute_stats(result_files)

    # Generate text report
    lines = []
    lines.append("=" * 60)
    lines.append("Evaluation Statistics")
    lines.append("=" * 60)

    lines.append(f"\n[Overall Statistics]")
    lines.append(f"  Total samples: {stats['total_samples']}")
    lines.append(f"  Error samples: {stats['error_count']}")
    lines.append(f"  Valid samples: {stats['valid_samples']}")

    lines.append(f"\n[Positive/Negative Sample Statistics]")
    lines.append(f"  Positive (consistent, pos_*): {stats['pos_count']}, correctly judged: {stats['pos_correct']}")
    lines.append(f"  Negative (inconsistent, neg_*): {stats['neg_count']}, correctly judged: {stats['neg_correct']}")

    lines.append(f"\n[Confusion Matrix]")
    lines.append(f"  TP (True Positive): {stats['TP']} - correctly identified inconsistency")
    lines.append(f"  FN (False Negative): {stats['FN']} - missed inconsistency (judged as consistent)")
    lines.append(f"  TN (True Negative): {stats['TN']} - correctly identified consistency")
    lines.append(f"  FP (False Positive): {stats['FP']} - false alarm (judged as inconsistent)")

    lines.append(f"\n[Recall, Precision, FPR]")
    lines.append(f"  Recall = TP/(TP+FN): {stats['recall']:.2%}")
    lines.append(f"  Precision = TP/(TP+FP): {stats['precision']:.2%}")
    lines.append(f"  FPR = FP/(FP+TN): {stats['fpr']:.2%}")
    lines.append(f"  Specificity = TN/(TN+FP): {stats['specificity']:.2%}")

    lines.append(f"\n[Exists Judgement Accuracy]")
    lines.append(f"  Correct: {stats['exists_correct']}")
    lines.append(f"  Incorrect: {stats['exists_incorrect']}")
    lines.append(f"  Accuracy: {stats['exists_accuracy']:.2%}")

    lines.append(f"\n[Classification Accuracy] (only for samples correctly judged as inconsistent)")
    lines.append(f"  Correctly judged as inconsistent (denominator): {stats['correct_inconsistent']}")
    lines.append(f"  Classification correct (numerator): {stats['classification_correct']}")
    lines.append(f"  Classification accuracy: {stats['classification_accuracy']:.2%}")

    if stats['category_stats']:
        lines.append(f"\n[Per-Category Classification Accuracy]")
        for category, cat_stats in sorted(stats['category_stats'].items()):
            if cat_stats['total'] > 0:
                acc = cat_stats['type_correct'] / cat_stats['total']
                lines.append(f"  {category}: {cat_stats['type_correct']}/{cat_stats['total']} = {acc:.2%}")

    lines.append(f"\n[Reasoning Metrics] (only for samples correctly judged as inconsistent, average, n={stats['correct_inconsistent']})")
    lines.append(f"  ROUGE-L: {stats['avg_rouge_l']:.2f}")
    lines.append(f"  METEOR: {stats['avg_meteor']:.2f}")
    lines.append(f"  BLEU-4: {stats['avg_bleu4']:.2f}")

    if stats['error_files']:
        lines.append(f"\n[Error Files]")
        for f in stats['error_files'][:10]:
            lines.append(f"  {f}")
        if len(stats['error_files']) > 10:
            lines.append(f"  ... and {len(stats['error_files']) - 10} more")

    lines.append("=" * 60)

    report = "\n".join(lines)

    # Print to stdout
    print(report)

    # Save to file if output_file is specified
    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nStatistics saved to: {args.output_file}")


if __name__ == "__main__":
    main()

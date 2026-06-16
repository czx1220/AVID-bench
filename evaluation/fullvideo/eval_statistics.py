#!/usr/bin/env python3
"""
Compute evaluation statistics from result.json files in a given results directory.
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


def get_all_result_files(results_dir: str) -> List[Path]:
    """Retrieve all *_result.json files from the given directory."""
    results_path = Path(results_dir)
    return sorted([f for f in results_path.glob('*_result.json')])


def compute_stats(result_files: List[Path]) -> Dict[str, Any]:
    """Compute evaluation statistics from a list of result files."""
    total = len(result_files)

    # Positive / negative sample counts
    pos_count = 0  # positive samples (consistent)
    pos_correct = 0  # correctly judged positive samples
    neg_count = 0  # negative samples (inconsistent)
    neg_correct = 0  # correctly judged negative samples

    # exists_correct=True sample counts
    exists_correct_count = 0
    exists_incorrect_count = 0

    # IoU metrics (only for exists_correct=True samples)
    all_ious = []
    iou_zero_count = 0
    pred_count = 0

    # Reasoning metrics (only for exists_correct=True samples)
    all_rouge_l = []
    all_meteor = []
    all_bleu4 = []
    reasoning_count = 0

    # SODA metrics (only for exists_correct=True samples)
    all_soda_m = []
    soda_count = 0

    # Error files
    error_count = 0
    error_files = []

    for filepath in result_files:
        try:
            data = load_result_json(filepath)

            # Determine file type (pos=consistent, neg=inconsistent)
            is_pos = filepath.name.startswith('pos_')
            is_neg = filepath.name.startswith('neg_')

            # Count positive / negative samples
            if is_pos:
                pos_count += 1
            elif is_neg:
                neg_count += 1

            # Count exists judgments
            if data.get('exists_correct') is True:
                exists_correct_count += 1
                # Correctly judged positive sample
                if is_pos:
                    pos_correct += 1
                # Correctly judged negative sample
                elif is_neg:
                    neg_correct += 1

                # Only compute mIoU etc. for negative samples (GT=Yes) with correct exists
                if is_neg:
                    pred_metrics = data.get('pred_metrics', [])
                    if pred_metrics:
                        for pm in pred_metrics:
                            iou = pm.get('iou', 0)
                            all_ious.append(iou)
                            pred_count += 1
                            if iou == 0:
                                iou_zero_count += 1

                            rouge_l = pm.get('rouge_l')
                            meteor = pm.get('meteor')
                            bleu4 = pm.get('bleu4')

                            if rouge_l is not None:
                                all_rouge_l.append(rouge_l)
                                reasoning_count += 1
                            if meteor is not None:
                                all_meteor.append(meteor)
                            if bleu4 is not None:
                                all_bleu4.append(bleu4)

                    # SODA metrics
                    soda_m = data.get('soda_m')
                    if soda_m is not None:
                        all_soda_m.append(soda_m)
                        soda_count += 1

            elif data.get('exists_correct') is False:
                exists_incorrect_count += 1

            # Count errors
            if data.get('error'):
                error_count += 1
                error_files.append(data.get('qa_id', filepath.name))

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            error_count += 1
            error_files.append(filepath.name)

    # Compute IoU metrics (only for exists_correct=True)
    r0_3 = sum(1 for iou in all_ious if iou >= 0.3) / len(all_ious) if all_ious else 0
    r0_5 = sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) if all_ious else 0
    r0_7 = sum(1 for iou in all_ious if iou >= 0.7) / len(all_ious) if all_ious else 0
    miou = sum(all_ious) / len(all_ious) if all_ious else 0

    # Compute average reasoning metrics
    avg_rouge_l = sum(all_rouge_l) / len(all_rouge_l) if all_rouge_l else 0
    avg_meteor = sum(all_meteor) / len(all_meteor) if all_meteor else 0
    avg_bleu4 = sum(all_bleu4) / len(all_bleu4) if all_bleu4 else 0

    # Compute average SODA metrics
    avg_soda_m = sum(all_soda_m) / len(all_soda_m) if all_soda_m else 0

    # Confusion matrix
    TP = neg_correct
    FN = neg_count - neg_correct
    TN = pos_correct
    FP = pos_count - pos_correct

    # Compute accuracy metrics
    valid_total = total - error_count
    exists_accuracy = exists_correct_count / valid_total if valid_total > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    valid_accuracy = (TP + TN) / valid_total if valid_total > 0 else 0

    return {
        'total_samples': total,
        'valid_samples': valid_total,
        'error_count': error_count,
        'exists_correct_count': exists_correct_count,
        'exists_incorrect_count': exists_incorrect_count,
        'exists_accuracy': exists_accuracy,
        # Positive / negative sample counts
        'pos_count': pos_count,
        'pos_correct': pos_correct,
        'neg_count': neg_count,
        'neg_correct': neg_correct,
        # Confusion matrix
        'TP': TP,
        'FN': FN,
        'TN': TN,
        'FP': FP,
        'recall': recall,
        'precision': precision,
        'fpr': fpr,
        'specificity': specificity,
        'valid_accuracy': valid_accuracy,
        # IoU metrics (only for exists_correct=True)
        'pred_count': pred_count,
        'iou_zero_count': iou_zero_count,
        'r0_3': r0_3,
        'r0_5': r0_5,
        'r0_7': r0_7,
        'miou': miou,
        # Reasoning metrics (only for exists_correct=True)
        'reasoning_count': reasoning_count,
        'avg_rouge_l': avg_rouge_l,
        'avg_meteor': avg_meteor,
        'avg_bleu4': avg_bleu4,
        # SODA metrics (only for exists_correct=True)
        'soda_count': soda_count,
        'avg_soda_m': avg_soda_m,
        # Error files
        'error_files': error_files
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation statistics from result.json files."
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Path to the directory containing *_result.json files."
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to save the statistics report. If not specified, only prints to stdout."
    )
    args = parser.parse_args()

    result_files = get_all_result_files(args.results_dir)
    print("Results directory:", args.results_dir)
    print(f"Found {len(result_files)} result.json files")

    stats = compute_stats(result_files)

    # Generate text report
    lines = []
    lines.append("=" * 60)
    lines.append("Evaluation Statistics (FullVideo)")
    lines.append("=" * 60)

    lines.append(f"\n[Overall Statistics]")
    lines.append(f"  Total samples: {stats['total_samples']}")
    lines.append(f"  Error samples: {stats['error_count']}")
    lines.append(f"  Valid samples: {stats['valid_samples']}")

    lines.append(f"\n[Exists Judgment]")
    lines.append(f"  Correct: {stats['exists_correct_count']}")
    lines.append(f"  Incorrect: {stats['exists_incorrect_count']}")
    lines.append(f"  Accuracy: {stats['exists_accuracy']:.2%}")

    lines.append(f"\n[Positive / Negative Sample Statistics]")
    lines.append(f"  Positive (consistent, pos_*): {stats['pos_count']}, correctly judged: {stats['pos_correct']}")
    lines.append(f"  Negative (inconsistent, neg_*): {stats['neg_count']}, correctly judged: {stats['neg_correct']}")

    lines.append(f"\n[Confusion Matrix]")
    lines.append(f"  TP (True Positive): {stats['TP']} - correctly identified inconsistency")
    lines.append(f"  FN (False Negative): {stats['FN']} - missed inconsistency")
    lines.append(f"  TN (True Negative): {stats['TN']} - correctly identified consistency")
    lines.append(f"  FP (False Positive): {stats['FP']} - falsely flagged inconsistency")

    lines.append(f"\n[Recall, Precision, FPR]")
    lines.append(f"  Recall: {stats['recall']:.2%}")
    lines.append(f"  Precision: {stats['precision']:.2%}")
    lines.append(f"  FPR (False Positive Rate): {stats['fpr']:.2%}")
    lines.append(f"  Specificity: {stats['specificity']:.2%}")
    lines.append(f"  Overall Accuracy: {stats['valid_accuracy']:.2%}")

    lines.append(f"\n[IoU Metrics] (only for exists_correct=True)")
    lines.append(f"  Total predictions: {stats['pred_count']}")
    lines.append(f"  IoU=0 count: {stats['iou_zero_count']}")
    lines.append(f"  R@0.3: {stats['r0_3']:.2%}")
    lines.append(f"  R@0.5: {stats['r0_5']:.2%}")
    lines.append(f"  R@0.7: {stats['r0_7']:.2%}")
    lines.append(f"  mIoU: {stats['miou']:.4f}")

    lines.append(f"\n[Reasoning Metrics] (only for exists_correct=True)")
    lines.append(f"  Count: {stats['reasoning_count']}")
    lines.append(f"  ROUGE-L (avg): {stats['avg_rouge_l']:.2f}")
    lines.append(f"  METEOR (avg): {stats['avg_meteor']:.2f}")
    lines.append(f"  BLEU-4 (avg): {stats['avg_bleu4']:.2f}")

    lines.append(f"\n[SODA Metrics] (only for exists_correct=True)")
    lines.append(f"  Count: {stats['soda_count']}")
    lines.append(f"  SODA-m (avg): {stats['avg_soda_m']:.4f}")

    if stats['error_files']:
        lines.append(f"\n[Error Files]")
        for f in stats['error_files'][:10]:
            lines.append(f"  {f}")
        if len(stats['error_files']) > 10:
            lines.append(f"  ... and {len(stats['error_files']) - 10} more")

    lines.append("=" * 60)

    report = "\n".join(lines)

    # Print to terminal
    print(report)

    # Save to file if output_file is specified
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nStatistics saved to: {args.output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate grounding accuracy (IoU) of predicted time intervals."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='Evaluate temporal grounding IoU accuracy')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to directory containing *_result.json files')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                        help='IoU thresholds for accuracy calculation (default: 0.3 0.5 0.7)')
    args = parser.parse_args()

    thresholds = args.thresholds

    # Store per-video results
    results = []

    # Iterate over all json files
    for fname in os.listdir(args.results_dir):
        if not fname.endswith('_result.json'):
            continue
        # Only evaluate negative (inconsistent) samples
        if not fname.startswith('neg_'):
            continue
        fpath = os.path.join(args.results_dir, fname)

        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Only evaluate videos where both GT and prediction indicate existence
        if data.get('ground_truth', {}).get('exists') != 'Yes':
            continue
        if data.get('parsed_answer', {}).get('exists') != 'Yes':
            continue

        # Ground truth interval count
        gt_count = len(data.get('ground_truth', {}).get('inconsistencies', []))

        # Predicted interval count (compatible with both field names)
        parsed = data.get('parsed_answer', {})
        pred_count = len(parsed.get('inconsistencies', []) or parsed.get('inconsistency_points', []))

        # Collect IoU values
        iou_list = [m.get('iou', 0) for m in data.get('pred_metrics', []) if m.get('iou') is not None]

        # Compute per-video accuracy at each threshold
        iou_stats = {}
        for thresh in thresholds:
            if len(iou_list) > 0:
                correct_count = sum(1 for iou in iou_list if iou >= thresh)
                accuracy = correct_count / len(iou_list)
            else:
                accuracy = 0.0
            iou_stats[thresh] = accuracy

        results.append({
            'qa_id': data.get('qa_id', fname),
            'gt_count': gt_count,
            'pred_count': pred_count,
            'iou_list': iou_list,
            'iou_stats': iou_stats
        })

    # Overall statistics
    total_videos = len(results)
    if total_videos == 0:
        print("No valid results found.")
        return

    total_gt_intervals = sum(r['gt_count'] for r in results)
    total_pred_intervals = sum(r['pred_count'] for r in results)

    avg_pred_count = total_pred_intervals / total_videos
    avg_gt_count = total_gt_intervals / total_videos

    print("=" * 50)
    print("Grounding Accuracy (IoU) Results")
    print("=" * 50)
    print(f"\nTotal videos: {total_videos}")
    print(f"Total predicted intervals: {total_pred_intervals}")
    print(f"Total ground truth intervals: {total_gt_intervals}")
    print(f"Avg GT intervals per video: {avg_gt_count:.4f}")
    print(f"Avg predicted intervals per video: {avg_pred_count:.4f}")
    print(f"Avg interval count difference: {avg_pred_count - avg_gt_count:.4f}")

    print(f"\nAccuracy at each IoU threshold (correct predictions / total predictions):")
    for thresh in thresholds:
        correct_count = sum(sum(1 for iou in r['iou_list'] if iou >= thresh) for r in results)
        total_pred = sum(r['pred_count'] for r in results)
        if total_pred > 0:
            accuracy = correct_count / total_pred
        else:
            accuracy = 0.0
        print(f"  IoU >= {thresh}: {accuracy:.4f} ({correct_count}/{total_pred})")


if __name__ == "__main__":
    main()

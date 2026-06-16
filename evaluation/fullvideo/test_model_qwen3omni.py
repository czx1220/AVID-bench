#!/usr/bin/env python3
"""
Video QA Evaluation Script for Qwen3-Omni (Batch Inference on Full Videos)

This script evaluates audio-visual inconsistency detection on full-length videos
using the Qwen3-Omni model with batch inference. It loads QA pairs from JSON
files, runs the model on corresponding full videos, parses model responses with
temporal span annotations, and computes metrics including existence accuracy,
temporal IoU, ROUGE-L, METEOR, BLEU-4, and SODA-m.

Usage:
    python test_model_qwen3omni.py --qa-dir ./data/qa --batch-size 2
"""

import os

import json
import re
import time
import argparse
import logging
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge_score not installed, ROUGE-L will be skipped")

# NLTK metrics
try:
    import nltk
    from nltk.translate.meteor_score import meteor_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not installed, METEOR and BLEU will be skipped")

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# ==================== Configuration ====================
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QA_DIR = "./data/qa"
OUTPUT_DIR = "./results"

USE_AUDIO_IN_VIDEO = True
BATCH_SIZE = 2  # Batch size, reduced for full-length videos

# Prompt with classification guide
CLASSIFICATION_GUIDE_EN = """
【8 Inconsistency Types (must choose from these)】

**Class 1 (Active Speaker - interview/dialogue, person speaking in video)**:
- TEMPORAL_SHIFT: Temporal offset, audio leads or lags behind video by 0.5-2s (e.g., see mouth moving but sound is delayed by 0.5s)
- LIP_SYNC: Lip sync mismatch, TTS generated voice doesn't match lip movement (e.g., video shows a man speaking but voice sounds like a woman)
- VOICE_IDENTITY: Voice identity conflict, speaker's voice changes abruptly (e.g., video shows an elderly person but voice is a child's)
- VOLUME_FLUCTUATION: Volume conflict, person is still but volume fluctuates (e.g., person stands still but voice sounds like it's moving closer and further)

**Class 2 (Voiceover - narration, no speaker in video)**:
- SEMANTIC_DIVERGENCE: Semantic inconsistency, TTS text contradicts video content (e.g., video shows food but narration talks about phones)
- BACKGROUND_CONFLICT: Background sound conflict, narration/music contradicts video scene (e.g., video shows office but background has bar music)

**Class 3 (Scenic - scenery/scene, no human voice)**:
- EMOTION_MISMATCH: Background music emotion mismatch, video is sad scene but has happy music (e.g., video shows funeral but music is joyful)
- BACKGROUND_SOUND: Background sound conflict, video is scene A but audio is scene B (e.g., video shows forest but has city traffic sounds; e.g., video shows chopping but no chopping sound)
"""

# Full-video specific question prompt
DEFAULT_QUESTION_EN_FULLVIDEO = f"""Please carefully watch this full video (not a short segment), analyze its audio and visual content, and determine whether there is audio-visual inconsistency.

{CLASSIFICATION_GUIDE_EN}

Please answer in the following exact format (each item on a new line):
1. Is there any inconsistency in this video? (Answer: Yes or No)
2. If Yes, for each inconsistency, provide: "from X.Xs to Y.Ys, reasoning" - each on a separate line. If No, write "N/A"

Example output format:
Yes
from 0.0s to 7.8s, The background music of sad feelings creates emotional conflict with the lively visual tone
from 15.5s to 20.3s, The sound of rain is injected but no rain is shown in the indoor scene
"""

# ==================== Data Structures ====================
@dataclass
class EvaluationResult:
    """Evaluation result for a single QA pair"""
    qa_id: str
    video_path: str
    model_answer: str
    parsed_answer: Dict[str, Any] = field(default_factory=dict)
    exists_correct: bool = False
    # Per-GT-segment metrics: [{iou, rouge_l, meteor, bleu4}, ...]
    pred_metrics: List[Dict[str, float]] = field(default_factory=list)
    soda_m: float = 0.0  # SODA-m metric
    error: Optional[str] = None


@dataclass
class AggregatedResults:
    """Aggregated evaluation results"""
    total: int = 0
    exists_accuracy: float = 0.0
    injection_type_accuracy: float = 0.0
    rouge_l_mean: float = 0.0
    meteor_mean: float = 0.0
    bleu4_mean: float = 0.0


# ==================== Qwen Local Model Wrapper ====================
class QwenLocalAPI:
    """Qwen3-Omni local model with batch inference support"""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model (only loads once)"""
        if self.model is None:
            print(f"Loading model from {self.model_path}...")
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            # Disable talker (no audio generation)
            self.model.disable_talker()
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
            print("Model loaded successfully!")
        return self.model, self.processor

    def chat_batch(self, video_paths: List[str], prompt: str = DEFAULT_QUESTION_EN_FULLVIDEO) -> List[str]:
        """Batch inference on multiple videos"""
        model, processor = self.load_model()

        # Build batch conversations
        conversations = []
        for video_path in video_paths:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt}
                    ],
                },
            ]
            conversations.append(conversation)

        # Process batch inputs
        text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text,
                           audio=audios,
                           images=images,
                           videos=videos,
                           return_tensors="pt",
                           padding=True,
                           use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        # Batch generation
        text_ids, _ = model.generate(**inputs,
                                      return_audio=False,
                                      thinker_return_dict_in_generate=True,
                                      use_audio_in_video=USE_AUDIO_IN_VIDEO)

        results = processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        return results


# ==================== Answer Parsing ====================
# ==================== Temporal Span Evaluation ====================
def parse_time_range(time_str: str) -> tuple:
    """Parse a time range string, returns (start, end)"""
    match = re.match(r'from\s+([\d.]+)s?\s+to\s+([\d.]+)s?', time_str, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def calculate_iou(span1: tuple, span2: tuple) -> float:
    """Calculate IoU between two temporal spans"""
    start1, end1 = span1
    start2, end2 = span2
    if start1 >= end1 or start2 >= end2:
        return 0.0

    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0.0


def find_best_match(pred_span: tuple, gt_spans: list) -> tuple:
    """Find the GT span with highest IoU for a predicted span"""
    best_iou = 0
    best_idx = -1
    for idx, gt_span in enumerate(gt_spans):
        iou = calculate_iou(pred_span, gt_span)
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx, best_iou


def calculate_soda_m(pred_inconsistencies: List[Dict], gt_inconsistencies: List[Dict]) -> float:
    """Calculate SODA-m metric"""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not pred_inconsistencies or not gt_inconsistencies:
        return 0.0

    n_pred = len(pred_inconsistencies)
    n_gt = len(gt_inconsistencies)

    # Parse time ranges
    pred_spans = []
    pred_reasonings = []
    for inc in pred_inconsistencies:
        start, end = parse_time_range(inc.get("time_range", ""))
        pred_spans.append((start, end))
        pred_reasonings.append(inc.get("reasoning", ""))

    gt_spans = []
    gt_points = []
    for inc in gt_inconsistencies:
        start, end = parse_time_range(inc.get("time_range", ""))
        gt_spans.append((start, end))
        gt_points.append(inc.get("inconsistency_point", ""))

    # Compute score matrix: f_sim = METEOR x IoU
    score_matrix = np.zeros((n_pred, n_gt))
    for i, pred_span in enumerate(pred_spans):
        for j, gt_span in enumerate(gt_spans):
            iou = calculate_iou(pred_span, gt_span)
            if iou > 0:
                meteor = calculate_meteor(gt_points[j], pred_reasonings[i])
                score_matrix[i, j] = meteor * iou

    # Hungarian algorithm for optimal matching
    if score_matrix.size > 0:
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        total_score = score_matrix[row_ind, col_ind].sum()
    else:
        total_score = 0.0

    # Compute F1
    if n_pred > 0:
        precision = total_score / n_pred
    else:
        precision = 0.0

    if n_gt > 0:
        recall = total_score / n_gt
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1


def parse_model_answer(answer: str) -> Dict[str, Any]:
    """Parse full-video model answer - time_range and reasoning as arrays"""
    result = {
        "exists": "",
        "inconsistencies": []  # Array format: [{"time_range": "...", "reasoning": "..."}, ...]
    }

    lines = answer.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    if not lines:
        return result

    # 1. First line: whether inconsistency exists (Yes/No)
    first_line = lines[0]
    if re.search(r'\bYes\b', first_line, re.IGNORECASE):
        result["exists"] = "Yes"
    elif re.search(r'\bNo\b', first_line, re.IGNORECASE):
        result["exists"] = "No"

    # 2. From the second line onward: time_range + reasoning array
    if len(lines) >= 2 and result["exists"] == "Yes":
        for line in lines[1:]:
            if line.lower() == "n/a":
                continue
            # Match "from X.Xs to Y.Ys, reasoning" format
            match = re.match(r'(from\s+[\d.]+s\s+to\s+[\d.]+s)\s*,\s*(.+)', line, re.IGNORECASE)
            if match:
                result["inconsistencies"].append({
                    "time_range": match.group(1),
                    "reasoning": match.group(2).strip()
                })

    return result


# ==================== Metric Computation ====================
def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    """Calculate ROUGE-L score (scaled to 0-100)"""
    if not ROUGE_AVAILABLE or not reference or not hypothesis:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure * 100
    except:
        return 0.0


def calculate_meteor(reference: str, hypothesis: str) -> float:
    """Calculate METEOR score (scaled to 0-100)"""
    if not NLTK_AVAILABLE or not reference or not hypothesis:
        return 0.0
    try:
        from nltk.tokenize import word_tokenize
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        return meteor_score([ref_tokens], hyp_tokens) * 100
    except:
        return 0.0


def calculate_bleu4(reference: str, hypothesis: str) -> float:
    """Calculate BLEU-4 score (scaled to 0-100)"""
    if not NLTK_AVAILABLE or not reference or not hypothesis:
        return 0.0
    try:
        from nltk.tokenize import word_tokenize
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())
        smoothing = SmoothingFunction().method1
        return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing) * 100
    except:
        return 0.0


def evaluate_single_qa(result: EvaluationResult, ground_truth: Dict, qa_id: str = "") -> EvaluationResult:
    """Evaluate a single QA pair - full-video version"""
    result.parsed_answer = parse_model_answer(result.model_answer)

    # Determine if this is a positive or negative sample
    is_negative = qa_id.startswith("neg_")

    # 1. Yes/No classification accuracy
    # neg_ -> Yes, pos_ -> No
    gt_exists = "Yes" if is_negative else "No"
    pred_exists = result.parsed_answer.get("exists", "")
    result.exists_correct = (gt_exists.upper() == pred_exists.upper())

    # 2. Temporal span + reasoning evaluation
    result.pred_metrics = []

    if is_negative and result.exists_correct:
        pred_inconsistencies = result.parsed_answer.get("inconsistencies", [])

        gt_inconsistencies = ground_truth.get("inconsistencies", [])
        gt_spans = []
        gt_points = []
        for inc in gt_inconsistencies:
            time_range = inc.get("time_range", "")
            start, end = parse_time_range(time_range)
            if start is not None:
                gt_spans.append((start, end))
                gt_points.append(inc.get("inconsistency_point", ""))

        # Initialize metrics for each GT span (default iou=0 means missed)
        gt_metrics = []
        for i in range(len(gt_spans)):
            gt_metrics.append({
                "iou": 0.0,
                "rouge_l": 0.0,
                "meteor": 0.0,
                "bleu4": 0.0
            })

        # For each predicted span, find best matching GT
        for pred_inc in pred_inconsistencies:
            pred_time_range = pred_inc.get("time_range", "")
            pred_reasoning = pred_inc.get("reasoning", "")
            start, end = parse_time_range(pred_time_range)

            if start is None or not pred_reasoning:
                continue

            pred_span = (start, end)
            best_idx, best_iou = find_best_match(pred_span, gt_spans)

            if best_idx >= 0 and best_idx < len(gt_metrics):
                if best_iou > gt_metrics[best_idx]["iou"]:
                    gt_point = gt_points[best_idx] if best_idx < len(gt_points) else ""
                    gt_metrics[best_idx]["iou"] = best_iou

                    if gt_point and pred_reasoning:
                        gt_metrics[best_idx]["rouge_l"] = calculate_rouge_l(gt_point, pred_reasoning)
                        gt_metrics[best_idx]["meteor"] = calculate_meteor(gt_point, pred_reasoning)
                        gt_metrics[best_idx]["bleu4"] = calculate_bleu4(gt_point, pred_reasoning)

        result.pred_metrics = gt_metrics

        # Compute SODA-m
        result.soda_m = calculate_soda_m(pred_inconsistencies, gt_inconsistencies)

    return result


def aggregate_results(results: List[EvaluationResult], qa_ids: List[str] = None) -> AggregatedResults:
    """Aggregate all results"""
    agg = AggregatedResults()
    agg.total = len(results)

    if agg.total == 0:
        return agg

    # Existence accuracy - all samples
    exists_correct = sum(1 for r in results if r.exists_correct)
    agg.exists_accuracy = exists_correct / agg.total

    # Injection type accuracy - negative samples only
    if qa_ids:
        negative_results = [r for r, qid in zip(results, qa_ids) if qid.startswith("neg_")]
        negative_count = len(negative_results)
        if negative_count > 0:
            type_correct = sum(1 for r in negative_results if r.injection_type_correct)
            agg.injection_type_accuracy = type_correct / negative_count
        else:
            agg.injection_type_accuracy = 0.0

        # Text similarity - negative samples only
        rouge_scores = [r.rouge_l_score for r in negative_results if r.rouge_l_score > 0]
        meteor_scores = [r.meteor_score for r in negative_results if r.meteor_score > 0]
        bleu_scores = [r.bleu4_score for r in negative_results if r.bleu4_score > 0]

        agg.rouge_l_mean = sum(rouge_scores) / negative_count if negative_count > 0 else 0.0
        agg.meteor_mean = sum(meteor_scores) / negative_count if negative_count > 0 else 0.0
        agg.bleu4_mean = sum(bleu_scores) / negative_count if negative_count > 0 else 0.0
    else:
        agg.injection_type_accuracy = 0.0

    return agg


# ==================== Main Pipeline ====================
def load_qa_files(qa_dir: str, limit: Optional[int] = None) -> List[Dict]:
    """Load QA files from directory"""
    qa_files = sorted(Path(qa_dir).glob("*.json"))
    if limit:
        qa_files = qa_files[:limit]

    qa_list = []
    for qa_file in qa_files:
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_list.append(json.load(f))
    return qa_list


def run_evaluation(
    qa_dir: str = QA_DIR,
    qa_files: Optional[str] = None,
    output_dir: str = OUTPUT_DIR,
    limit: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
):
    """Run evaluation pipeline"""
    api = QwenLocalAPI()

    # Load QA files
    if qa_files:
        file_paths = [f.strip() for f in qa_files.split(',')]
        qa_list = []
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                qa_list.append(json.load(f))
        print(f"Loaded {len(qa_list)} specific QA files")
    else:
        qa_list = load_qa_files(qa_dir, limit)
    print(f"Loaded {len(qa_list)} QA files, batch_size={batch_size}")

    os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(output_dir, "eval.log")
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    logging.getLogger('nltk').setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)

    results = []
    qa_ids = []

    # Preload model
    api.load_model()

    # Batch processing
    for batch_start in range(0, len(qa_list), batch_size):
        batch_end = min(batch_start + batch_size, len(qa_list))
        batch_qa = qa_list[batch_start:batch_end]

        # Collect pending QA pairs (skip already completed ones)
        pending_qa = []
        pending_indices = []
        for i, qa in enumerate(batch_qa):
            qa_id = qa['qa_id']
            result_file = os.path.join(output_dir, f"{qa_id}_result.json")
            if os.path.exists(result_file):
                logger.info(f"Skipped (already exists): {qa_id}")
                with open(result_file, 'r', encoding='utf-8') as f:
                    saved_result = json.load(f)
                result = EvaluationResult(
                    qa_id=qa_id,
                    video_path=qa.get("video_path", ""),
                    model_answer=saved_result.get("model_answer", ""),
                    parsed_answer=saved_result.get("parsed_answer", {}),
                    exists_correct=saved_result.get("exists_correct", False),
                    pred_metrics=saved_result.get("pred_metrics", []),
                    soda_m=saved_result.get("soda_m", 0.0),
                    error=saved_result.get("error")
                )
                results.append(result)
                qa_ids.append(qa_id)
            else:
                pending_qa.append(qa)
                pending_indices.append(batch_start + i)

        if not pending_qa:
            continue

        # Batch inference
        video_paths = [qa["video_path"] for qa in pending_qa]
        print(f"Processing batch {batch_start+1}-{batch_end}/{len(qa_list)}: {[qa['qa_id'] for qa in pending_qa]}")

        try:
            model_answers = api.chat_batch(video_paths, DEFAULT_QUESTION_EN_FULLVIDEO)

            # Save results
            for i, (qa, answer) in enumerate(zip(pending_qa, model_answers)):
                qa_id = qa['qa_id']
                result_file = os.path.join(output_dir, f"{qa_id}_result.json")

                result = EvaluationResult(
                    qa_id=qa_id,
                    video_path=qa["video_path"],
                    model_answer=answer
                )

                ground_truth = qa.get("ground_truth", {})
                result = evaluate_single_qa(result, ground_truth, qa_id)

                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "qa_id": result.qa_id,
                        "ground_truth": qa.get("ground_truth", {}),
                        "model_answer": result.model_answer,
                        "parsed_answer": result.parsed_answer,
                        "exists_correct": result.exists_correct,
                        "pred_metrics": result.pred_metrics,
                        "soda_m": result.soda_m,
                        "error": result.error,
                    }, f, ensure_ascii=False, indent=2)

                logger.info(f"Completed: {qa_id}")
                results.append(result)
                qa_ids.append(qa_id)

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            for qa in pending_qa:
                qa_id = qa['qa_id']
                result_file = os.path.join(output_dir, f"{qa_id}_result.json")
                result = EvaluationResult(
                    qa_id=qa_id,
                    video_path=qa["video_path"],
                    model_answer="",
                    error=str(e)
                )
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "qa_id": result.qa_id,
                        "ground_truth": qa.get("ground_truth", {}),
                        "model_answer": result.model_answer,
                        "error": result.error,
                    }, f, ensure_ascii=False, indent=2)
                results.append(result)
                qa_ids.append(qa_id)

        # Progress report
        if len(results) % 20 == 0:
            current_acc = sum(1 for r in results if r.exists_correct) / len(results)
            print(f"  Progress: {len(results)}/{len(qa_list)}, Exists Acc: {current_acc:.2%}")

    # Aggregate results
    positive_count = sum(1 for qid in qa_ids if qid.startswith("pos_"))
    negative_count = sum(1 for qid in qa_ids if qid.startswith("neg_"))
    exists_correct_count = sum(1 for r in results if r.exists_correct)
    type_correct_count = sum(1 for r, qid in zip(results, qa_ids) if qid.startswith("neg_") and r.injection_type_correct)

    negative_results = [r for r, qid in zip(results, qa_ids) if qid.startswith("neg_")]
    rouge_l_sum = sum(r.rouge_l_score for r in negative_results)
    meteor_sum = sum(r.meteor_score for r in negative_results)
    bleu4_sum = sum(r.bleu4_score for r in negative_results)

    summary = {
        "total": len(results),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "exists_correct_count": exists_correct_count,
        "type_correct_count": type_correct_count,
        "rouge_l_sum": rouge_l_sum,
        "meteor_sum": meteor_sum,
        "bleu4_sum": bleu4_sum,
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Total QA: {len(results)}")
    print(f"Exists Acc: {exists_correct_count}/{len(results)} = {exists_correct_count/len(results):.2%}")
    if negative_count > 0:
        print(f"Type Acc: {type_correct_count}/{negative_count} = {type_correct_count/negative_count:.2%}")
        print(f"ROUGE-L: {rouge_l_sum/negative_count:.2f}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Video QA Evaluation - Qwen Local (Batch)")
    parser.add_argument("--qa-dir", type=str, default=QA_DIR, help="QA directory")
    parser.add_argument("--qa-files", type=str, default=None, help="Specific QA files (comma-separated paths)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for inference")

    args = parser.parse_args()

    run_evaluation(
        qa_dir=args.qa_dir,
        qa_files=args.qa_files,
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
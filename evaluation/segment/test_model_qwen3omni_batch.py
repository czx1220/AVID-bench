#!/usr/bin/env python3
"""
Video QA Evaluation Script for Qwen3-Omni (Batch Inference on Segments)

This script evaluates audio-visual inconsistency detection on video segments
using the Qwen3-Omni model with batch inference. It loads QA pairs from JSON
files, runs the model on corresponding video segments, parses model responses,
and computes metrics including existence accuracy, type accuracy, ROUGE-L,
METEOR, and BLEU-4.

Usage:
    python test_model_qwen3omni_batch.py --qa-dir ./data/qa --batch-size 16
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
BATCH_SIZE = 16  # Batch size, adjustable via --batch-size argument

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

DEFAULT_QUESTION_EN = f"""Please carefully watch this video, analyze its audio and visual content, and determine whether there is audio-visual inconsistency.

Please return in the following exact format:
1. Is there inconsistency: (Yes/No) - Judge independently based on video content
2. Inconsistency type: (If Yes, choose one from the following 8 types. If "No", leave empty):

{CLASSIFICATION_GUIDE_EN}

3. Inconsistency point description: (if inconsistency exists, describe the specific inconsistency between visual and audio; if "No", describe why you think the audio and video are consistent)"""

# ==================== Data Structures ====================
@dataclass
class EvaluationResult:
    """Evaluation result for a single QA pair"""
    qa_id: str
    video_path: str
    model_answer: str
    parsed_answer: Dict[str, str] = field(default_factory=dict)
    exists_correct: bool = False
    injection_type_correct: bool = False
    rouge_l_score: float = 0.0
    meteor_score: float = 0.0
    bleu4_score: float = 0.0
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

    def chat_batch(self, video_paths: List[str], prompt: str = DEFAULT_QUESTION_EN) -> List[str]:
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
def parse_model_answer(answer: str) -> Dict[str, str]:
    """Parse model answer, extract 3 fields"""
    result = {
        "exists": "",
        "injection_type": "",
        "inconsistency_point": ""
    }

    # 1. Whether inconsistency exists (Yes/No)
    exists_match = re.search(r'1\.\s*Is there inconsistency[：:]\s*(Yes|No)', answer, re.IGNORECASE)
    if exists_match:
        result["exists"] = "Yes" if exists_match.group(1).lower() == "yes" else "No"

    # 2. Inconsistency type
    type_match = re.search(r'2\.\s*Inconsistency type[：:]\s*([A-Z_]+)', answer)
    if type_match:
        result["injection_type"] = type_match.group(1).strip()

    # 3. Inconsistency point description
    point_match = re.search(r'3\.\s*Inconsistency point description[：:]\s*(.+)', answer, re.DOTALL)
    if point_match:
        result["inconsistency_point"] = point_match.group(1).strip()

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
    """Evaluate a single QA pair"""
    result.parsed_answer = parse_model_answer(result.model_answer)

    # Determine if this is a positive or negative sample
    is_negative = qa_id.startswith("neg_")

    # 1. Existence accuracy - computed for all samples
    gt_exists = ground_truth.get("exists", "No")
    pred_exists = result.parsed_answer.get("exists", "")
    result.exists_correct = (gt_exists.upper() == pred_exists.upper())

    # 2. Injection type accuracy - only computed for negative samples
    if is_negative:
        gt_type = ground_truth.get("injection_type", "")
        pred_type = result.parsed_answer.get("injection_type", "")
        result.injection_type_correct = (gt_type.upper() == pred_type.upper())
    else:
        result.injection_type_correct = False  # Not counted for positive samples

    # 3. Text similarity - only computed for negative samples
    if is_negative:
        gt_point = ground_truth.get("inconsistency_point", "")
        pred_point = result.parsed_answer.get("inconsistency_point", "")
        if gt_point and pred_point:
            result.rouge_l_score = calculate_rouge_l(gt_point, pred_point)
            result.meteor_score = calculate_meteor(gt_point, pred_point)
            result.bleu4_score = calculate_bleu4(gt_point, pred_point)

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
                    injection_type_correct=saved_result.get("injection_type_correct", False),
                    rouge_l_score=saved_result.get("rouge_l_score", 0.0),
                    meteor_score=saved_result.get("meteor_score", 0.0),
                    bleu4_score=saved_result.get("bleu4_score", 0.0),
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
            model_answers = api.chat_batch(video_paths, DEFAULT_QUESTION_EN)

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
                        "injection_type_correct": result.injection_type_correct,
                        "rouge_l_score": result.rouge_l_score,
                        "meteor_score": result.meteor_score,
                        "bleu4_score": result.bleu4_score,
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
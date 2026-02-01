"""
Hallucination Evaluation Metrics for Sa2VA-Struct

This module provides three quantitative metrics for evaluating attention hallucination:
1. DCR (Distractor Confusion Rate): Measures confusion with same-class distractors
2. SRA (Spatial Reasoning Accuracy): Evaluates spatial relationship understanding
3. ADS (Attention Drift Score): Quantifies attention deviation from target

Usage:
    python hallucination_eval.py <model_path> --metric dcr --dataset refcocog
    python hallucination_eval.py <model_path> --metric sra --dataset refcocog
    python hallucination_eval.py <model_path> --metric all --dataset refcocog
"""

import argparse
import copy
import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from pycocotools import mask as _mask_utils

from transformers import AutoModel, AutoTokenizer

# Import from existing evaluation infrastructure
try:
    from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
    from dataset import RESDataset
except ImportError:
    from .utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
    from .dataset import RESDataset


# =============================================================================
# Spatial Keywords for SRA Evaluation
# =============================================================================

SPATIAL_KEYWORDS = {
    'positional': [
        'left', 'right', 'above', 'below', 'behind', 'in front', 'front of',
        'top', 'bottom', 'upper', 'lower', 'over', 'under', 'beneath'
    ],
    'relational': [
        'next to', 'beside', 'between', 'near', 'close to', 'far from',
        'adjacent', 'opposite', 'facing', 'touching'
    ],
    'ordinal': [
        'first', 'second', 'third', 'fourth', 'fifth', 'last',
        '1st', '2nd', '3rd', '4th', '5th', 'leftmost', 'rightmost'
    ],
    'size_comparative': [
        'larger', 'smaller', 'bigger', 'biggest', 'smallest', 'largest',
        'taller', 'shorter', 'wider', 'narrower', 'longer'
    ],
}


# =============================================================================
# Utility Functions
# =============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to RLE format."""
    rle = _mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode()
    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    """Convert RLE to binary mask."""
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode()
    return _mask_utils.decode(rle)


def classify_spatial_category(text: str) -> Optional[str]:
    """Classify referring expression into spatial category."""
    text_lower = text.lower()
    for category, keywords in SPATIAL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    return None


# =============================================================================
# Distractor Confusion Rate (DCR)
# =============================================================================

class DistractorConfusionEvaluator:
    """
    Evaluates Distractor Confusion Rate (DCR).
    
    DCR measures how often the model incorrectly segments same-class distractors
    instead of the target object.
    
    Lower DCR = Better (less hallucination)
    """
    
    def __init__(self, dataset, coco_annotations_path: Optional[str] = None):
        """
        Args:
            dataset: RESDataset instance
            coco_annotations_path: Path to COCO annotations for category info
        """
        self.dataset = dataset
        self.coco_annotations_path = coco_annotations_path
        self._build_category_index()
    
    def _build_category_index(self):
        """Build index mapping image_id to objects by category."""
        self.image_objects = defaultdict(lambda: defaultdict(list))
        
        # Access underlying RefCOCO data
        if hasattr(self.dataset, 'refer') or hasattr(self.dataset, 'data_list'):
            data_list = getattr(self.dataset, 'data_list', [])
            for item in data_list:
                img_id = item.get('img_id') or item.get('image_id')
                cat_id = item.get('category_id')
                ref_id = item.get('ref_id')
                mask = item.get('gt_mask') or item.get('mask')
                
                if img_id and cat_id:
                    self.image_objects[img_id][cat_id].append({
                        'ref_id': ref_id,
                        'mask': mask
                    })
    
    def get_distractor_masks(self, sample: dict) -> List[np.ndarray]:
        """
        Get masks of same-category objects that are NOT the target.
        
        Args:
            sample: Dataset sample with 'img_id', 'category_id', 'ref_id', 'gt_mask'
        
        Returns:
            List of distractor masks (same category, different instance)
        """
        img_id = sample.get('img_id') or sample.get('image_id')
        cat_id = sample.get('category_id')
        ref_id = sample.get('ref_id')
        
        if not img_id or not cat_id:
            return []
        
        distractor_masks = []
        for obj in self.image_objects[img_id][cat_id]:
            if obj['ref_id'] != ref_id and obj['mask'] is not None:
                mask = obj['mask']
                if isinstance(mask, dict):  # RLE format
                    mask = rle_to_mask(mask)
                distractor_masks.append(mask)
        
        return distractor_masks
    
    def compute_dcr(
        self, 
        predictions: List[dict], 
        samples: List[dict],
        iou_threshold: float = 0.3
    ) -> Dict[str, float]:
        """
        Compute Distractor Confusion Rate.
        
        Args:
            predictions: List of {'prediction_mask': np.ndarray}
            samples: List of dataset samples
            iou_threshold: IoU threshold to consider as "confused"
        
        Returns:
            Dictionary with DCR metrics
        """
        total_confused = 0
        total_samples_with_distractors = 0
        distractor_ious = []
        
        for pred, sample in zip(predictions, samples):
            pred_mask = pred.get('prediction_mask')
            if pred_mask is None:
                continue
            
            distractor_masks = self.get_distractor_masks(sample)
            
            if len(distractor_masks) == 0:
                continue
            
            total_samples_with_distractors += 1
            
            # Compute max IoU with any distractor
            max_distractor_iou = max([
                compute_iou(pred_mask, dm) for dm in distractor_masks
            ])
            distractor_ious.append(max_distractor_iou)
            
            if max_distractor_iou > iou_threshold:
                total_confused += 1
        
        dcr = total_confused / total_samples_with_distractors if total_samples_with_distractors > 0 else 0
        mean_distractor_iou = np.mean(distractor_ious) if distractor_ious else 0
        
        return {
            'dcr': dcr * 100,  # Percentage
            'mean_distractor_iou': mean_distractor_iou * 100,
            'n_samples_with_distractors': total_samples_with_distractors,
            'n_confused': total_confused,
        }


# =============================================================================
# Spatial Reasoning Accuracy (SRA)
# =============================================================================

class SpatialReasoningEvaluator:
    """
    Evaluates Spatial Reasoning Accuracy (SRA).
    
    SRA measures the model's ability to understand spatial relationships
    like "left of", "behind", "second from right", etc.
    
    Higher SRA = Better (more accurate spatial understanding)
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Args:
            iou_threshold: IoU threshold to consider prediction as correct
        """
        self.iou_threshold = iou_threshold
    
    def filter_spatial_samples(
        self, 
        dataset, 
        texts: List[str]
    ) -> Tuple[List[int], List[str], List[str]]:
        """
        Filter samples containing spatial keywords.
        
        Args:
            dataset: Dataset instance
            texts: List of referring expressions
        
        Returns:
            Tuple of (indices, categories, texts) for spatial samples
        """
        indices = []
        categories = []
        filtered_texts = []
        
        for idx, text in enumerate(texts):
            category = classify_spatial_category(text)
            if category is not None:
                indices.append(idx)
                categories.append(category)
                filtered_texts.append(text)
        
        return indices, categories, filtered_texts
    
    def compute_sra(
        self,
        predictions: List[dict],
        gt_masks: List[np.ndarray],
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Compute Spatial Reasoning Accuracy by category.
        
        Args:
            predictions: List of {'prediction_mask': np.ndarray}
            gt_masks: List of ground truth masks
            categories: List of spatial categories
        
        Returns:
            Dictionary with SRA metrics per category and overall
        """
        results = {cat: {'correct': 0, 'total': 0} for cat in SPATIAL_KEYWORDS}
        all_ious = []
        
        for pred, gt_mask, category in zip(predictions, gt_masks, categories):
            pred_mask = pred.get('prediction_mask')
            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask)
            
            iou = compute_iou(pred_mask, gt_mask)
            all_ious.append(iou)
            
            results[category]['total'] += 1
            if iou >= self.iou_threshold:
                results[category]['correct'] += 1
        
        # Compute per-category accuracy
        sra_by_category = {}
        for cat, stats in results.items():
            if stats['total'] > 0:
                sra_by_category[f'sra_{cat}'] = (stats['correct'] / stats['total']) * 100
                sra_by_category[f'n_{cat}'] = stats['total']
        
        # Overall SRA
        total_correct = sum(r['correct'] for r in results.values())
        total_samples = sum(r['total'] for r in results.values())
        overall_sra = (total_correct / total_samples * 100) if total_samples > 0 else 0
        
        return {
            'overall_sra': overall_sra,
            'mean_iou': np.mean(all_ious) * 100 if all_ious else 0,
            'n_spatial_samples': total_samples,
            **sra_by_category
        }


# =============================================================================
# Attention Drift Score (ADS)
# =============================================================================

class AttentionDriftEvaluator:
    """
    Evaluates Attention Drift Score (ADS).
    
    ADS measures how much attention drifts away from the target region.
    Requires access to attention weights from the model.
    
    Lower ADS = Better (attention focuses on target)
    """
    
    def __init__(self):
        pass
    
    def compute_ads_from_attention(
        self,
        attention_weights: torch.Tensor,
        gt_masks: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Attention Drift Score from attention weights.
        
        Args:
            attention_weights: (B, num_heads, HW, Nt) attention weights
            gt_masks: (B, H, W) ground truth masks
        
        Returns:
            Dictionary with ADS metrics
        """
        B, num_heads, HW, Nt = attention_weights.shape
        H = W = int(math.sqrt(HW))
        
        # Average attention across heads and text tokens
        spatial_attn = attention_weights.mean(dim=(1, 3))  # (B, HW)
        spatial_attn = spatial_attn.view(B, H, W)
        
        # Resize GT mask to attention resolution
        gt_masks_resized = F.interpolate(
            gt_masks.unsqueeze(1).float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # Normalize attention for each sample
        spatial_attn = spatial_attn / (spatial_attn.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        # Compute attention within target region
        target_attn = (spatial_attn * gt_masks_resized).sum(dim=(1, 2))
        
        # ADS = 1 - (attention in target region)
        ads_per_sample = 1 - target_attn
        
        return {
            'ads': ads_per_sample.mean().item(),
            'ads_std': ads_per_sample.std().item(),
            'target_attention_ratio': target_attn.mean().item(),
        }
    
    def compute_ads_from_predictions(
        self,
        predictions: List[dict],
        gt_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Approximate ADS using prediction quality as proxy.
        
        When attention weights are not available, we use mask quality
        as a proxy for attention focus.
        
        Args:
            predictions: List of {'prediction_mask': np.ndarray}
            gt_masks: List of ground truth masks
        
        Returns:
            Dictionary with proxy ADS metrics
        """
        ious = []
        boundary_overlaps = []
        
        for pred, gt in zip(predictions, gt_masks):
            pred_mask = pred.get('prediction_mask')
            if pred_mask is None:
                ious.append(0)
                continue
            
            iou = compute_iou(pred_mask, gt)
            ious.append(iou)
        
        # Proxy ADS: inverse of mean IoU (higher IoU = lower drift)
        mean_iou = np.mean(ious) if ious else 0
        proxy_ads = 1 - mean_iou
        
        return {
            'proxy_ads': proxy_ads,
            'mean_iou': mean_iou,
        }


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hallucination Evaluation for Sa2VA-Struct")
    parser.add_argument("model_path", help="Path to HuggingFace model")
    parser.add_argument(
        "--metric",
        choices=["dcr", "sra", "ads", "all"],
        default="all",
        help="Which metric to compute"
    )
    parser.add_argument(
        "--dataset",
        choices=["refcoco", "refcoco_plus", "refcocog"],
        default="refcocog",
        help="Dataset to evaluate on"
    )
    parser.add_argument("--split", default="val", help="Dataset split")
    parser.add_argument("--image-folder", default=None, help="Path to COCO images")
    parser.add_argument("--data-path", default=None, help="Path to RefCOCO data")
    parser.add_argument("--work-dir", default="./work_dirs/hallucination_eval", help="Output directory")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


DATASETS_ATTRIBUTES = {
    "refcoco": {"splitBy": "unc", "dataset_name": "refcoco"},
    "refcoco_plus": {"splitBy": "unc", "dataset_name": "refcoco_plus"},
    "refcocog": {"splitBy": "umd", "dataset_name": "refcocog"},
}


def main():
    args = parse_args()
    
    # Setup distributed
    if args.launcher != "none":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
    else:
        rank, world_size = 0, 1
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = (
        AutoModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load dataset
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    image_folder = args.image_folder or os.environ.get("COCO2014_TRAIN", "")
    data_path = args.data_path or os.environ.get("REFCOCO_DATA", "")
    
    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=dataset_info["dataset_name"],
        data_path=data_path,
        split=args.split,
    )
    
    # Collect predictions
    print("Running inference...")
    predictions = []
    samples = []
    texts = []
    
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)
    per_rank_ids = range(
        per_rank_samples * rank,
        min(n_samples, per_rank_samples * (rank + 1))
    )
    
    for idx in tqdm.tqdm(per_rank_ids, desc="Inference"):
        data_batch = dataset[idx]
        sample_info = {
            'img_id': data_batch.get('img_id'),
            'gt_mask': data_batch.get('gt_masks'),
            'category_id': data_batch.get('category_id'),
            'ref_id': data_batch.get('ref_id'),
        }
        samples.append(sample_info)
        
        text_list = data_batch.get('text', [])
        if isinstance(text_list, str):
            text_list = [text_list]
        texts.extend(text_list)
        
        # Run inference
        del data_batch['img_id'], data_batch['gt_masks']
        data_batch_copy = copy.deepcopy(data_batch)
        del data_batch_copy['text']
        
        pred_masks = []
        for text in text_list:
            _batch = copy.deepcopy(data_batch_copy)
            _batch['text'] = text
            
            with torch.no_grad():
                output = model.predict_forward(**_batch, tokenizer=tokenizer)
                pred_mask = output.get('prediction_masks', [])
            
            if len(pred_mask) > 0:
                pred_masks.append(pred_mask[0])
            else:
                pred_masks.append(None)
        
        predictions.append({
            'prediction_masks': pred_masks,
            'prediction_mask': pred_masks[0] if pred_masks else None,
        })
    
    # Gather results
    if args.launcher != "none":
        predictions = collect_results_cpu(predictions, n_samples)
        samples = collect_results_cpu(samples, n_samples)
    
    # Compute metrics
    if rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        results = {}
        
        # DCR Evaluation
        if args.metric in ["dcr", "all"]:
            print("\n" + "="*50)
            print("Computing Distractor Confusion Rate (DCR)...")
            print("="*50)
            dcr_evaluator = DistractorConfusionEvaluator(dataset)
            dcr_results = dcr_evaluator.compute_dcr(predictions, samples)
            results['dcr'] = dcr_results
            print(f"DCR: {dcr_results['dcr']:.2f}%")
            print(f"Mean Distractor IoU: {dcr_results['mean_distractor_iou']:.2f}%")
            print(f"Samples with Distractors: {dcr_results['n_samples_with_distractors']}")
        
        # SRA Evaluation
        if args.metric in ["sra", "all"]:
            print("\n" + "="*50)
            print("Computing Spatial Reasoning Accuracy (SRA)...")
            print("="*50)
            sra_evaluator = SpatialReasoningEvaluator()
            indices, categories, spatial_texts = sra_evaluator.filter_spatial_samples(
                dataset, texts
            )
            
            if len(indices) > 0:
                spatial_predictions = [predictions[i] for i in indices]
                spatial_gt_masks = [samples[i]['gt_mask'] for i in indices]
                
                sra_results = sra_evaluator.compute_sra(
                    spatial_predictions, spatial_gt_masks, categories
                )
                results['sra'] = sra_results
                
                print(f"Overall SRA: {sra_results['overall_sra']:.2f}%")
                print(f"Spatial Samples: {sra_results['n_spatial_samples']}")
                for cat in SPATIAL_KEYWORDS:
                    key = f'sra_{cat}'
                    if key in sra_results:
                        print(f"  {cat}: {sra_results[key]:.2f}% (n={sra_results.get(f'n_{cat}', 0)})")
            else:
                print("No spatial samples found in dataset.")
        
        # ADS Evaluation (Proxy)
        if args.metric in ["ads", "all"]:
            print("\n" + "="*50)
            print("Computing Attention Drift Score (ADS)...")
            print("="*50)
            ads_evaluator = AttentionDriftEvaluator()
            gt_masks = [s['gt_mask'] for s in samples if s['gt_mask'] is not None]
            valid_preds = [p for p, s in zip(predictions, samples) if s['gt_mask'] is not None]
            
            ads_results = ads_evaluator.compute_ads_from_predictions(valid_preds, gt_masks)
            results['ads'] = ads_results
            print(f"Proxy ADS: {ads_results['proxy_ads']:.4f}")
            print(f"Mean IoU: {ads_results['mean_iou']:.4f}")
        
        # Save results
        output_path = os.path.join(
            args.work_dir, 
            f"hallucination_eval_{args.dataset}_{args.split}.json"
        )
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        
        # Print summary table
        print("\n" + "="*60)
        print("HALLUCINATION EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.dataset} ({args.split})")
        print("-"*60)
        if 'dcr' in results:
            print(f"Distractor Confusion Rate (DCR) ↓: {results['dcr']['dcr']:.2f}%")
        if 'sra' in results:
            print(f"Spatial Reasoning Accuracy (SRA) ↑: {results['sra']['overall_sra']:.2f}%")
        if 'ads' in results:
            print(f"Attention Drift Score (ADS) ↓: {results['ads']['proxy_ads']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()

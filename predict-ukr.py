"""
U-BDD++ Evaluation with pre-trained CLIP - Modified for UKR TIF tile-based inference (1024x1024)
"""
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
import torchvision.transforms as T
import numpy as np
import cv2
from tqdm import tqdm
import os
from PIL import Image
import glob
import math
from sklearn.metrics import f1_score, roc_auc_score
import json

from segment_anything import SamPredictor, sam_model_registry
import clip

# ...existing imports...
from models.clipmlp.clipmlp import clip_prediction_ensemble, CONTRASTIVE_PROMPTS
from models.dino.util.slconfig import SLConfig
from models.dino.models.registry import MODULE_BUILD_FUNCS
from models.dino.util import box_ops
from utils.filters import preliminary_filter
from utils.utils import pixel_f1_iou

# Constants - Updated for 1024x1024 tiles
TILE_SIZE = 1024
DINO_TEXT_PROMPT = "building"
DAMAGE_DICT_BGR = [[0, 0, 0], [70, 172, 0], [0, 140, 253]]


def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU) for binary masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def calculate_metrics(pred_mask, gt_mask=None):
    """Calculate IoU, F1, and AUROC metrics"""
    metrics = {}
    
    # If no ground truth is provided, calculate basic statistics
    if gt_mask is None:
        metrics['damage_ratio'] = (pred_mask == 2).sum() / pred_mask.size
        metrics['building_ratio'] = ((pred_mask == 1) | (pred_mask == 2)).sum() / pred_mask.size
        return metrics
    
    # Convert to binary masks for different classes
    pred_building = (pred_mask > 0).astype(np.uint8)  # Any building (damaged or undamaged)
    pred_damage = (pred_mask == 2).astype(np.uint8)   # Only damaged buildings
    
    gt_building = (gt_mask > 0).astype(np.uint8)      # Any building (damaged or undamaged)
    gt_damage = (gt_mask == 2).astype(np.uint8)       # Only damaged buildings
    
    # Calculate IoU for building detection
    metrics['building_iou'] = calculate_iou(pred_building, gt_building)
    
    # Calculate IoU for damage detection
    metrics['damage_iou'] = calculate_iou(pred_damage, gt_damage)
    
    # Calculate F1 scores
    if gt_building.sum() > 0:
        pred_flat = pred_building.flatten()
        gt_flat = gt_building.flatten()
        metrics['building_f1'] = f1_score(gt_flat, pred_flat, average='binary')
    else:
        metrics['building_f1'] = 0.0
    
    if gt_damage.sum() > 0:
        pred_flat = pred_damage.flatten()
        gt_flat = gt_damage.flatten()
        metrics['damage_f1'] = f1_score(gt_flat, pred_flat, average='binary')
    else:
        metrics['damage_f1'] = 0.0
    
    # Calculate AUROC (only if we have both classes)
    try:
        if len(np.unique(gt_building)) > 1:
            metrics['building_auroc'] = roc_auc_score(gt_building.flatten(), pred_building.flatten())
        else:
            metrics['building_auroc'] = 0.5
            
        if len(np.unique(gt_damage)) > 1:
            metrics['damage_auroc'] = roc_auc_score(gt_damage.flatten(), pred_damage.flatten())
        else:
            metrics['damage_auroc'] = 0.5
    except ValueError:
        metrics['building_auroc'] = 0.5
        metrics['damage_auroc'] = 0.5
    
    return metrics


def calculate_detailed_metrics(pred_mask, gt_mask=None):
    """Calculate detailed IoU, F1, and AUROC metrics"""
    metrics = {}
    
    # If no ground truth is provided, calculate pixel-based metrics
    if gt_mask is None:
        # Basic statistics
        metrics['damage_ratio'] = float((pred_mask == 2).sum()) / float(pred_mask.size)
        metrics['building_ratio'] = float(((pred_mask == 1) | (pred_mask == 2)).sum()) / float(pred_mask.size)
        
        # Calculate pixel-based IoU metrics
        pixel_iou_metrics = calculate_pixel_based_iou(pred_mask)
        
        # Use pixel-based IoU as a substitute for the missing ground truth IoU
        metrics['building_iou'] = pixel_iou_metrics['building_coverage']
        metrics['damage_iou'] = pixel_iou_metrics['damage_building_iou']
        metrics['building_f1'] = 2 * pixel_iou_metrics['building_coverage'] / (1 + pixel_iou_metrics['building_coverage'])
        metrics['damage_f1'] = 2 * pixel_iou_metrics['damage_precision'] / (1 + pixel_iou_metrics['damage_precision'])
        metrics['building_auroc'] = 0.5  # Default value when no ground truth
        metrics['damage_auroc'] = 0.5    # Default value when no ground truth
        
        # Add new pixel-based metrics
        metrics.update(pixel_iou_metrics)
        return metrics
    
    # Convert to binary masks for different classes
    pred_building = (pred_mask > 0).astype(np.uint8)  # Any building (damaged or undamaged)
    pred_damage = (pred_mask == 2).astype(np.uint8)   # Only damaged buildings
    
    gt_building = (gt_mask > 0).astype(np.uint8)      # Any building (damaged or undamaged)
    gt_damage = (gt_mask == 2).astype(np.uint8)       # Only damaged buildings
    
    # Calculate IoU for building detection
    metrics['building_iou'] = calculate_iou(pred_building, gt_building)
    
    # Calculate IoU for damage detection
    metrics['damage_iou'] = calculate_iou(pred_damage, gt_damage)
    
    # Calculate F1 scores
    if gt_building.sum() > 0:
        pred_flat = pred_building.flatten()
        gt_flat = gt_building.flatten()
        metrics['building_f1'] = f1_score(gt_flat, pred_flat, average='binary')
    else:
        metrics['building_f1'] = 0.0
    
    if gt_damage.sum() > 0:
        pred_flat = pred_damage.flatten()
        gt_flat = gt_damage.flatten()
        metrics['damage_f1'] = f1_score(gt_flat, pred_flat, average='binary')
    else:
        metrics['damage_f1'] = 0.0
    
    # Calculate AUROC (only if we have both classes)
    try:
        if len(np.unique(gt_building)) > 1:
            metrics['building_auroc'] = roc_auc_score(gt_building.flatten(), pred_building.flatten())
        else:
            metrics['building_auroc'] = 0.5
            
        if len(np.unique(gt_damage)) > 1:
            metrics['damage_auroc'] = roc_auc_score(gt_damage.flatten(), pred_damage.flatten())
        else:
            metrics['damage_auroc'] = 0.5
    except ValueError:
        metrics['building_auroc'] = 0.5
        metrics['damage_auroc'] = 0.5
    
    # Add basic statistics
    metrics['damage_ratio'] = float((pred_mask == 2).sum()) / float(pred_mask.size)
    metrics['building_ratio'] = float(((pred_mask == 1) | (pred_mask == 2)).sum()) / float(pred_mask.size)
    
    return metrics


def calculate_pixel_based_iou(pred_mask):
    """Calculate pixel-based IoU metrics from the prediction mask without ground truth"""
    
    # Extract different regions
    damaged_pixels = (pred_mask == 2)
    undamaged_pixels = (pred_mask == 1)
    background_pixels = (pred_mask == 0)
    all_buildings = damaged_pixels | undamaged_pixels
    
    # Calculate total pixels
    total_pixels = pred_mask.size
    total_damaged = damaged_pixels.sum()
    total_undamaged = undamaged_pixels.sum()
    total_buildings = all_buildings.sum()
    total_background = background_pixels.sum()
    
    # Calculate IoU between damaged and all buildings
    # IoU = intersection / union = damaged / all buildings
    damage_building_iou = 0.0
    if total_buildings > 0:
        damage_building_iou = total_damaged / total_buildings
    
    # Calculate IoU between undamaged and all buildings
    # IoU = intersection / union = undamaged / all buildings
    undamaged_building_iou = 0.0
    if total_buildings > 0:
        undamaged_building_iou = total_undamaged / total_buildings
    
    # Calculate damage-to-undamaged ratio
    damage_undamaged_ratio = 0.0
    if total_undamaged > 0:
        damage_undamaged_ratio = total_damaged / total_undamaged
    
    # Calculate building coverage
    building_coverage = total_buildings / total_pixels
    
    # Calculate precision-like metrics (what percentage of detected buildings are damaged)
    damage_precision = 0.0
    if total_buildings > 0:
        damage_precision = total_damaged / total_buildings
    
    return {
        'damage_building_iou': float(damage_building_iou),
        'undamaged_building_iou': float(undamaged_building_iou),
        'damage_undamaged_ratio': float(damage_undamaged_ratio),
        'building_coverage': float(building_coverage),
        'damage_precision': float(damage_precision),
        'total_damaged_pixels': int(total_damaged),
        'total_undamaged_pixels': int(total_undamaged),
        'total_building_pixels': int(total_buildings),
        'total_background_pixels': int(total_background)
    }

# Add these functions after the save_detailed_metrics function (around line 540)

def load_ground_truth_mask(tif_path, gt_dir):
    """Load ground truth mask for a given TIF file"""
    # Extract city name from TIF filename
    base_name = os.path.basename(tif_path)
    
    # Try to extract city name (assuming format: CityName_pre/post_disaster_ortho.tif)
    city_name = None
    for city in ['Bucha', 'Chernihiv', 'Kerson', 'Mykolaiv']:
        if base_name.lower().startswith(city.lower()):
            city_name = city
            break
    
    if city_name is None:
        return None
    
    # Load ground truth mask
    gt_mask_path = os.path.join(gt_dir, f"{city_name}_ground_truth.npy")
    if os.path.exists(gt_mask_path):
        return np.load(gt_mask_path)
    
    # Try alternative naming conventions
    gt_mask_path = os.path.join(gt_dir, f"{city_name.lower()}_ground_truth.npy")
    if os.path.exists(gt_mask_path):
        return np.load(gt_mask_path)
        
    # Try loading PNG format
    gt_mask_path = os.path.join(gt_dir, f"{city_name}_ground_truth.png")
    if os.path.exists(gt_mask_path):
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        return gt_mask
    
    return None


def calculate_metrics_with_ground_truth(pred_mask, gt_mask):
    """Calculate comprehensive building-level metrics using ground truth - Enhanced Version"""
    return calculate_building_level_metrics(pred_mask, gt_mask)


def calculate_weighted_metrics_with_ground_truth(all_results):
    """Calculate weighted IoU, F1, and AUROC across all images using ground truth masks"""
    
    # Separate results with and without ground truth
    results_with_gt = [r for r in all_results if r.get('has_ground_truth', False)]
    results_without_gt = [r for r in all_results if not r.get('has_ground_truth', False)]
    
    weighted_metrics = {
        'total_files': len(all_results),
        'files_with_gt': len(results_with_gt),
        'files_without_gt': len(results_without_gt)
    }
    
    if len(results_with_gt) == 0:
        # No ground truth available - use existing logic
        return calculate_weighted_metrics(all_results)
    
    # Aggregate pixel-level predictions and ground truth for weighted metrics
    all_pred_building = []
    all_gt_building = []
    all_pred_damage = []
    all_gt_damage = []
    all_weights = []
    
    # Totals for weighted averages
    total_building_pixels_pred = 0
    total_damage_pixels_pred = 0
    total_building_pixels_gt = 0
    total_damage_pixels_gt = 0
    total_image_pixels = 0
    weighted_damage_ratio_sum = 0
    weighted_building_ratio_sum = 0
    weighted_building_iou_sum = 0
    weighted_damage_iou_sum = 0
    weighted_building_f1_sum = 0
    weighted_damage_f1_sum = 0
    weighted_building_auroc_sum = 0
    weighted_damage_auroc_sum = 0
    
    # Process results with ground truth
    for result in results_with_gt:
        metrics = result['metrics']
        
        # Extract image dimensions and calculate weight
        image_size_str = result['image_size']
        width, height = map(int, image_size_str.split('x'))
        image_pixels = width * height
        weight = image_pixels
        all_weights.append(weight)
        
        # Accumulate weighted metrics
        building_iou = metrics.get('building_iou', 0)
        damage_iou = metrics.get('damage_iou', 0)
        building_f1 = metrics.get('building_f1', 0)
        damage_f1 = metrics.get('damage_f1', 0)
        building_auroc = metrics.get('building_auroc', 0.5)
        damage_auroc = metrics.get('damage_auroc', 0.5)
        damage_ratio = metrics.get('damage_ratio', 0)
        building_ratio = metrics.get('building_ratio', 0)
        
        weighted_damage_ratio_sum += damage_ratio * weight
        weighted_building_ratio_sum += building_ratio * weight
        weighted_building_iou_sum += building_iou * weight
        weighted_damage_iou_sum += damage_iou * weight
        weighted_building_f1_sum += building_f1 * weight
        weighted_damage_f1_sum += damage_f1 * weight
        weighted_building_auroc_sum += building_auroc * weight
        weighted_damage_auroc_sum += damage_auroc * weight
        
        # Accumulate pixel counts
        total_building_pixels_pred += result['total_undamaged_pixels'] + result['total_damaged_pixels']
        total_damage_pixels_pred += result['total_damaged_pixels']
        total_image_pixels += image_pixels
        
        # For building-level pixel aggregation, create binary masks
        pred_building_pixels = result['total_undamaged_pixels'] + result['total_damaged_pixels']
        pred_damage_pixels = result['total_damaged_pixels']
        
        # Ground truth pixel counts (approximated from ratios)
        gt_damage_ratio = metrics.get('gt_damage_ratio', damage_ratio)
        gt_building_ratio = metrics.get('gt_building_ratio', building_ratio)
        gt_building_pixels = int(gt_building_ratio * image_pixels)
        gt_damage_pixels = int(gt_damage_ratio * image_pixels)
        
        total_building_pixels_gt += gt_building_pixels
        total_damage_pixels_gt += gt_damage_pixels
        
        # Create weighted pixel-level arrays for global F1 calculation
        # Weight each pixel by the image weight to ensure proper weighting
        pixel_weight = weight / image_pixels if image_pixels > 0 else 0
        
        # Building predictions and ground truth
        building_pred_array = np.concatenate([
            np.ones(pred_building_pixels, dtype=np.uint8),
            np.zeros(image_pixels - pred_building_pixels, dtype=np.uint8)
        ])
        building_gt_array = np.concatenate([
            np.ones(gt_building_pixels, dtype=np.uint8),
            np.zeros(image_pixels - gt_building_pixels, dtype=np.uint8)
        ])
        
        # Damage predictions and ground truth
        damage_pred_array = np.concatenate([
            np.ones(pred_damage_pixels, dtype=np.uint8),
            np.zeros(image_pixels - pred_damage_pixels, dtype=np.uint8)
        ])
        damage_gt_array = np.concatenate([
            np.ones(gt_damage_pixels, dtype=np.uint8),
            np.zeros(image_pixels - gt_damage_pixels, dtype=np.uint8)
        ])
        
        # Add to global arrays with proper weighting
        all_pred_building.extend(building_pred_array)
        all_gt_building.extend(building_gt_array)
        all_pred_damage.extend(damage_pred_array)
        all_gt_damage.extend(damage_gt_array)
   

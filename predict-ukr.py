"""
U-BDD++ Evaluation with pre-trained CLIP - Modified for UKR TIF tile-based inference (512x512)
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

# Constants - Updated for 512x512 tiles
TILE_SIZE = 512
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


class TIFTileDataset(Dataset):
    """Dataset class for TIF image tile-based inference with 512x512 tiles"""
    
    def __init__(self, tif_file_path, tile_size=TILE_SIZE, overlap=64):
        self.tif_file_path = tif_file_path
        self.tile_size = tile_size
        self.overlap = overlap  # Increased overlap for better edge handling
        
        # Load the full TIF image
        try:
            self.full_image = Image.open(tif_file_path).convert('RGB')
            self.original_width, self.original_height = self.full_image.size
            print(f"Loaded TIF image: {self.original_width}x{self.original_height}")
        except Exception as e:
            raise ValueError(f"Error loading TIF file {tif_file_path}: {e}")
        
        # Calculate tile grid for 512x512 tiles with overlap
        step_size = tile_size - overlap
        self.tiles_x = max(1, math.ceil((self.original_width - overlap) / step_size))
        self.tiles_y = max(1, math.ceil((self.original_height - overlap) / step_size))
        self.total_tiles = self.tiles_x * self.tiles_y
        
        print(f"Will create {self.tiles_x}x{self.tiles_y} = {self.total_tiles} tiles of size {tile_size}x{tile_size} (overlap: {overlap}px)")
        
        # Create tile coordinates
        self.tile_coords = []
        for y in range(self.tiles_y):
            for x in range(self.tiles_x):
                # Calculate tile boundaries
                start_x = x * step_size
                start_y = y * step_size
                end_x = min(start_x + tile_size, self.original_width)
                end_y = min(start_y + tile_size, self.original_height)
                
                # Ensure we don't go below 0
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                
                # For edge tiles, adjust to get full tile size when possible
                if end_x - start_x < tile_size and start_x > 0:
                    start_x = max(0, end_x - tile_size)
                if end_y - start_y < tile_size and start_y > 0:
                    start_y = max(0, end_y - tile_size)
                    
                # Recalculate end coordinates
                end_x = min(start_x + tile_size, self.original_width)
                end_y = min(start_y + tile_size, self.original_height)
                
                self.tile_coords.append({
                    'tile_x': x,
                    'tile_y': y,
                    'start_x': start_x,
                    'start_y': start_y,
                    'end_x': end_x,
                    'end_y': end_y,
                    'width': end_x - start_x,
                    'height': end_y - start_y
                })
        
        # DINO transform - resize to 512x512 for consistent processing
        self.normalize = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Transform for original image (without normalization but with resize)
        self.to_tensor = T.Compose([
            T.Resize((512, 512), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ])
    
    def __len__(self):
        return self.total_tiles
    
    def __getitem__(self, idx):
        tile_info = self.tile_coords[idx]
        
        # Extract tile from full image
        tile_image = self.full_image.crop((
            tile_info['start_x'],
            tile_info['start_y'],
            tile_info['end_x'],
            tile_info['end_y']
        ))
        
        # Store original tile size before resizing
        original_tile_size = tile_image.size
        
        # Resize tile to exactly 512x512 for processing
        tile_image_resized = tile_image.resize((self.tile_size, self.tile_size), Image.Resampling.BILINEAR)
        
        # Apply transforms
        pre_image = self.normalize(tile_image_resized)
        pre_image_original = self.to_tensor(tile_image_resized)
        
        return {
            'pre_image': pre_image,
            'pre_image_original': pre_image_original,
            'post_image_original': pre_image_original,
            'tile_info': tile_info,
            'tile_idx': idx,
            'file_name': os.path.basename(self.tif_file_path),
            'original_tile_size': original_tile_size
        }
    
    def get_image_info(self):
        """Return information about the original image"""
        return {
            'width': self.original_width,
            'height': self.original_height,
            'tiles_x': self.tiles_x,
            'tiles_y': self.tiles_y,
            'total_tiles': self.total_tiles,
            'tile_size': self.tile_size,
            'overlap': self.overlap
        }


class TIFInferenceDataset(Dataset):
    """Dataset class for multiple TIF files"""
    
    def __init__(self, tif_folder_path):
        self.tif_folder_path = tif_folder_path
        
        # Find all TIF files in the folder
        self.tif_files = glob.glob(os.path.join(tif_folder_path, "*.tif")) + \
                        glob.glob(os.path.join(tif_folder_path, "*.tiff"))
        
        if len(self.tif_files) == 0:
            raise ValueError(f"No TIF files found in {tif_folder_path}")
        
        print(f"Found {len(self.tif_files)} TIF files for inference")
    
    def __len__(self):
        return len(self.tif_files)
    
    def __getitem__(self, idx):
        return self.tif_files[idx]


def merge_tile_predictions_with_overlap(tile_results, image_info, output_path_base):
    """Merge individual tile predictions with overlap handling - optimized for 256x256 tiles"""
    
    # Initialize full-size prediction arrays
    full_width = image_info['width']
    full_height = image_info['height']
    overlap = image_info.get('overlap', 0)
    
    # Create full-size prediction mask and weight mask for overlap handling
    full_pred_mask = np.zeros((full_height, full_width), dtype=np.float32)
    weight_mask = np.zeros((full_height, full_width), dtype=np.float32)
    
    # Process each tile result
    for tile_result in tile_results:
        tile_info = tile_result['tile_info']
        pred_mask = tile_result['pred_mask']
        
        # Convert to numpy array if it's a tensor and ensure float32 type
        if hasattr(pred_mask, 'cpu'):
            pred_mask = pred_mask.cpu().numpy()
        if hasattr(pred_mask, 'detach'):
            pred_mask = pred_mask.detach().numpy()
        pred_mask = np.asarray(pred_mask, dtype=np.float32)
        
        # Get original tile dimensions
        original_tile_width = tile_info['width']
        original_tile_height = tile_info['height']
        
        # Resize prediction mask back to original tile size
        if pred_mask.shape != (original_tile_height, original_tile_width):
            pred_mask_resized = cv2.resize(
                pred_mask, 
                (original_tile_width, original_tile_height), 
                interpolation=cv2.INTER_NEAREST  # Use nearest neighbor to preserve class labels
            )
        else:
            pred_mask_resized = pred_mask
        
        # Create weight map for this tile (higher weight in center, lower at edges for overlap)
        tile_weight = np.ones((original_tile_height, original_tile_width), dtype=np.float32)
        
        if overlap > 0:
            # Create distance-based weights (higher in center, lower at edges) - ensure NumPy arrays
            y_coords = np.arange(original_tile_height, dtype=np.float32)
            x_coords = np.arange(original_tile_width, dtype=np.float32)
            y_indices, x_indices = np.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Distance from edges (all NumPy operations)
            dist_from_left = x_indices
            dist_from_right = (original_tile_width - 1 - x_indices)
            dist_from_top = y_indices
            dist_from_bottom = (original_tile_height - 1 - y_indices)
            
            # Minimum distance from any edge
            edge_distance = np.minimum(
                np.minimum(dist_from_left, dist_from_right),
                np.minimum(dist_from_top, dist_from_bottom)
            )
            
            # Weight based on distance from edge (within overlap region)
            overlap_factor = min(overlap // 2, min(original_tile_height, original_tile_width) // 4)
            if overlap_factor > 0:
                tile_weight = np.minimum(1.0, edge_distance / float(overlap_factor))
        
        # Ensure all arrays are NumPy float32
        pred_mask_resized = np.asarray(pred_mask_resized, dtype=np.float32)
        tile_weight = np.asarray(tile_weight, dtype=np.float32)
        
        # Place tile prediction in the full image with weighted averaging
        start_y, start_x = int(tile_info['start_y']), int(tile_info['start_x'])
        end_y, end_x = start_y + original_tile_height, start_x + original_tile_width
        
        # Ensure indices are within bounds
        start_y = max(0, start_y)
        start_x = max(0, start_x)
        end_y = min(full_height, end_y)
        end_x = min(full_width, end_x)
        
        # Adjust tile data if needed for boundary cases
        actual_height = end_y - start_y
        actual_width = end_x - start_x
        
        if actual_height != original_tile_height or actual_width != original_tile_width:
            pred_mask_resized = pred_mask_resized[:actual_height, :actual_width]
            tile_weight = tile_weight[:actual_height, :actual_width]
        
        # Accumulate weighted predictions (all numpy operations)
        full_pred_mask[start_y:end_y, start_x:end_x] += pred_mask_resized * tile_weight
        weight_mask[start_y:end_y, start_x:end_x] += tile_weight
    
    # Normalize by weights to get final prediction
    weight_mask[weight_mask == 0] = 1  # Avoid division by zero
    full_pred_mask = full_pred_mask / weight_mask
    
    # Round to nearest integer class
    full_pred_mask = np.round(full_pred_mask).astype(np.uint8)
    
    # Create colored prediction mask
    color_mask = np.zeros((full_height, full_width, 3), dtype=np.uint8)
    for i in range(3):
        color_mask[full_pred_mask == i] = DAMAGE_DICT_BGR[i]
    
    # Save results
    base_name = os.path.splitext(os.path.basename(output_path_base))[0]
    
    # Save colored prediction
    color_output_path = f"{output_path_base}_prediction_color.png"
    cv2.imwrite(color_output_path, color_mask)
    
    # Save raw prediction mask
    mask_output_path = f"{output_path_base}_prediction_mask.png"
    cv2.imwrite(mask_output_path, full_pred_mask)
    
    return full_pred_mask, color_mask


# ...existing functions remain the same...
def build_dino_model(args):
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)  # type: ignore
    return model, criterion, postprocessors


def load_dino_model(model_config_path, model_checkpoint_path, device="cpu"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model, criterion, postprocessors = build_dino_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, criterion, postprocessors


def get_dino_output(model, image, dino_threshold, postprocessors, device):
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None])

    outputs = postprocessors["bbox"](outputs, torch.Tensor([[1.0, 1.0]]).to(device))[0]
    scores = outputs["scores"]
    boxes = box_ops.box_xyxy_to_cxcywh(outputs["boxes"])
    select_mask = scores > dino_threshold

    pred_dict = {
        "boxes": boxes[select_mask],
        "scores": scores[select_mask],
        "labels": [DINO_TEXT_PROMPT] * len(scores[select_mask]),
    }
    return pred_dict


def process_single_tile(
    batch,
    dino_model,
    dino_postprocessors,
    sam_predictor,
    clip_text,
    clip_model,
    clip_preprocess,
    clip_min_patch_size,
    clip_img_padding,
    dino_threshold,
    device
):
    """Process a single 512x512 tile and return prediction results"""
    
    # Get DINO predictions
    output = get_dino_output(
        dino_model,
        batch["pre_image"],
        dino_threshold,
        dino_postprocessors,
        device=device,
    )
    boxes = output["boxes"].detach().cpu()  # cxcywh
    logits = output["scores"].detach().cpu()
    phrases = output["labels"]
    
    # Convert boxes to xyxy format (scale by 512 for 512x512 tiles)
    boxes = box_convert(boxes * TILE_SIZE, "cxcywh", "xyxy")
    
    # SAM prediction for all bounding boxes
    source_image = (
        (batch["pre_image_original"].permute(1, 2, 0) * 255)
        .numpy()
        .astype(np.uint8)
    )
    sam_predictor.set_image(source_image)
    
    if len(boxes) == 0:
        # No bounding box predictions
        masks = torch.zeros(
            (1, 1, source_image.shape[0], source_image.shape[1]),
            dtype=torch.uint8,
        ).to(device)
        predictions = []
    else:
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes, source_image.shape[:2]
        ).to(device)
        
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
        
        # CLIP Prediction for each bounding box (adjusted for 512x512 tiles)
        predictions = []
        for bbox in boxes.tolist():
            # Crop out the image given bboxes
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # Apply the image buffer (restored to original values for 512x512 tiles)
            image_buffer_x = (
                (clip_min_patch_size - w) / 2.0
                if w < clip_min_patch_size
                else clip_img_padding
            )
            image_buffer_y = (
                (clip_min_patch_size - h) / 2.0
                if h < clip_min_patch_size
                else clip_img_padding
            )
            
            # Add padding for prediction
            x1_pad = max(int(x1 - image_buffer_x), 0)
            y1_pad = max(int(y1 - image_buffer_y), 0)
            x2_pad = min(int(x2 + image_buffer_x), TILE_SIZE)
            y2_pad = min(int(y2 + image_buffer_y), TILE_SIZE)
            
            # Ensure we have a valid patch
            if x2_pad > x1_pad and y2_pad > y1_pad:
                pre_building_patch = T.ToPILImage()(
                    batch["pre_image_original"][:, y1_pad:y2_pad, x1_pad:x2_pad]
                )
                pre_building_patch_clip = (
                    clip_preprocess(pre_building_patch).unsqueeze(0).to(device)
                )
                
                # For TIF inference, use the same image for both pre and post
                post_building_patch_clip = pre_building_patch_clip
                
                pred = clip_prediction_ensemble(
                    clip_model,
                    pre_building_patch_clip,
                    post_building_patch_clip,
                    clip_text,
                )
                predictions.append(pred + 1)
            else:
                # If patch is too small, default to undamaged
                predictions.append(1)
    
    # Create prediction mask
    if len(predictions) == 0:
        pred_mask = masks[0].squeeze(0)
    else:
        # 0: background, 1: undamaged, 2: damaged
        pred_mask = (
            (
                masks.mul(
                    torch.tensor(predictions).to(device).reshape(-1, 1, 1, 1)
                )
            )
            .max(dim=0)[0]
            .squeeze(0)
        )
    
    # Convert prediction mask to numpy array
    pred_mask_np = pred_mask.cpu().numpy()
    
    return {
        'pred_mask': pred_mask_np,  # Ensure this is always a NumPy array
        'num_detections': len(boxes),
        'num_damaged': (pred_mask_np == 2).sum(),
        'num_undamaged': (pred_mask_np == 1).sum(),
        'boxes': boxes.tolist(),
        'scores': logits.tolist(),
        'predictions': predictions,
        'tile_info': batch['tile_info']
    }


# Updated inference function for 512x512 tile-based processing with metrics
def ubdd_plusplus_tile_inference(
    dino_model,
    dino_postprocessors,
    sam_predictor,
    clip_text,
    clip_model,
    clip_preprocess,
    clip_min_patch_size,
    clip_img_padding,
    dino_threshold,
    save_annotations,
    tif_files,
    device,
    output_dir,
    overlap=64  # Increased overlap for 512x512 tiles
):
    """Run 512x512 tile-based inference on TIF images and save results with metrics"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    overall_metrics = {
        'total_files': 0,
        'total_tiles': 0,
        'total_detections': 0,
        'total_damaged_pixels': 0,
        'total_undamaged_pixels': 0,
        'average_damage_ratio': 0.0,
        'average_building_ratio': 0.0
    }
    
    for tif_file in tqdm(tif_files, desc="Processing TIF files"):
        print(f"\nProcessing: {os.path.basename(tif_file)}")
        
        # Create tile dataset for this TIF file (512x512 tiles)
        tile_dataset = TIFTileDataset(tif_file, tile_size=TILE_SIZE, overlap=overlap)
        tile_dataloader = DataLoader(tile_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        image_info = tile_dataset.get_image_info()
        tile_results = []
        
        # Process each tile
        with tqdm(tile_dataloader, desc="Processing 512x512 tiles", leave=False) as tile_pbar:
            for batch in tile_pbar:
                # Extract single item from batch
                single_batch = {}
                for k, v in batch.items():
                    if k == 'tile_info':
                        # tile_info is a list of dicts, extract the first dict
                        single_batch[k] = v[0] if isinstance(v, list) and len(v) > 0 else v
                    elif isinstance(v, (list, tuple)) and len(v) > 0:
                        # For tensors and other data, get the first item
                        single_batch[k] = v[0]
                    elif hasattr(v, 'squeeze'):
                        # For tensors, remove batch dimension
                        single_batch[k] = v.squeeze(0)
                    else:
                        single_batch[k] = v
                
                # Process single tile
                tile_result = process_single_tile(
                    single_batch,
                    dino_model,
                    dino_postprocessors,
                    sam_predictor,
                    clip_text,
                    clip_model,
                    clip_preprocess,
                    clip_min_patch_size,
                    clip_img_padding,
                    dino_threshold,
                    device
                )
                
                tile_results.append(tile_result)
                
                tile_pbar.set_postfix(
                    tile=f"{len(tile_results)}/{image_info['total_tiles']}",
                    detections=tile_result['num_detections'],
                    refresh=False
                )
        
        # Merge tiles into full image with overlap handling
        base_name = os.path.splitext(os.path.basename(tif_file))[0]
        output_path_base = os.path.join(output_dir, base_name)
        
        print(f"Merging {len(tile_results)} 512x512 tiles with overlap handling...")
        full_pred_mask, color_mask = merge_tile_predictions_with_overlap(
            tile_results, image_info, output_path_base
        )
        
        # Calculate metrics for this image
        metrics = calculate_metrics(full_pred_mask)
        
        # Save original image for reference (downscaled if too large)
        original_image = Image.open(tif_file).convert('RGB')
        if max(original_image.size) > 4096:  # Downscale very large images for reference
            original_image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        
        original_output_path = f"{output_path_base}_original.png"
        original_image.save(original_output_path)
        
        # Calculate summary statistics
        total_detections = sum(tile['num_detections'] for tile in tile_results)
        total_damaged = (full_pred_mask == 2).sum()
        total_undamaged = (full_pred_mask == 1).sum()
        
        # Store results with metrics
        result = {
            'file_name': os.path.basename(tif_file),
            'image_size': f"{image_info['width']}x{image_info['height']}",
            'tile_size': f"{TILE_SIZE}x{TILE_SIZE}",
            'num_tiles': len(tile_results),
            'overlap': image_info['overlap'],
            'total_detections': total_detections,
            'total_damaged_pixels': int(total_damaged),
            'total_undamaged_pixels': int(total_undamaged),
            'damage_percentage': float(total_damaged) / (image_info['width'] * image_info['height']) * 100,
            'metrics': metrics,
            'tile_results': tile_results
        }
        all_results.append(result)
        
        # Update overall metrics
        overall_metrics['total_files'] += 1
        overall_metrics['total_tiles'] += len(tile_results)
        overall_metrics['total_detections'] += total_detections
        overall_metrics['total_damaged_pixels'] += int(total_damaged)
        overall_metrics['total_undamaged_pixels'] += int(total_undamaged)
        overall_metrics['average_damage_ratio'] += metrics.get('damage_ratio', 0)
        overall_metrics['average_building_ratio'] += metrics.get('building_ratio', 0)
        
        print(f"Completed {os.path.basename(tif_file)}: "
              f"{total_detections} detections, "
              f"{total_damaged} damaged pixels, "
              f"{total_undamaged} undamaged pixels, "
              f"Damage ratio: {metrics.get('damage_ratio', 0):.4f}")
    
    # Calculate average metrics
    if overall_metrics['total_files'] > 0:
        overall_metrics['average_damage_ratio'] /= overall_metrics['total_files']
        overall_metrics['average_building_ratio'] /= overall_metrics['total_files']
    
    # Save summary results with metrics
    summary_path = os.path.join(output_dir, "tile_inference_summary_512.txt")
    with open(summary_path, 'w') as f:
        f.write("TIF 512x512 Tile-based Inference Summary with Metrics\n")
        f.write("====================================================\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write(f"  Total files processed: {overall_metrics['total_files']}\n")
        f.write(f"  Total tiles processed: {overall_metrics['total_tiles']}\n")
        f.write(f"  Total detections: {overall_metrics['total_detections']}\n")
        f.write(f"  Average damage ratio: {overall_metrics['average_damage_ratio']:.4f}\n")
        f.write(f"  Average building ratio: {overall_metrics['average_building_ratio']:.4f}\n\n")
        
        # Per-file results
        f.write("PER-FILE RESULTS:\n")
        f.write("-" * 50 + "\n")
        for result in all_results:
            f.write(f"File: {result['file_name']}\n")
            f.write(f"  Image size: {result['image_size']}\n")
            f.write(f"  Tile size: {result['tile_size']}\n")
            f.write(f"  Overlap: {result['overlap']}px\n")
            f.write(f"  Number of tiles: {result['num_tiles']}\n")
            f.write(f"  Total detections: {result['total_detections']}\n")
            f.write(f"  Damaged pixels: {result['total_damaged_pixels']}\n")
            f.write(f"  Undamaged pixels: {result['total_undamaged_pixels']}\n")
            f.write(f"  Damage percentage: {result['damage_percentage']:.2f}%\n")
            
            metrics = result['metrics']
            f.write(f"  Metrics:\n")
            f.write(f"    Damage ratio: {metrics.get('damage_ratio', 0):.4f}\n")
            f.write(f"    Building ratio: {metrics.get('building_ratio', 0):.4f}\n")
            f.write("\n")
    
    # Save detailed metrics as JSON
    metrics_path = os.path.join(output_dir, "metrics_512.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'overall_metrics': overall_metrics,
            'per_file_metrics': [
                {
                    'file_name': result['file_name'],
                    'metrics': result['metrics'],
                    'summary': {
                        'total_detections': result['total_detections'],
                        'damage_percentage': result['damage_percentage'],
                        'num_tiles': result['num_tiles']
                    }
                }
                for result in all_results
            ]
        }, f, indent=2)
    
    print(f"\n512x512 tile-based inference completed! Results saved to: {output_dir}")
    print(f"Processed {len(all_results)} TIF images")
    print(f"Average damage ratio: {overall_metrics['average_damage_ratio']:.4f}")
    print(f"Average building ratio: {overall_metrics['average_building_ratio']:.4f}")
    return all_results


def get_args():
    parser = argparse.ArgumentParser(description="TIF Image 512x512 Tile-based Inference with U-BDD++ and Metrics")
    parser.add_argument(
        "--tif-folder-path",
        "-tfp",
        type=str,
        default=r"C:\SAI\IA\D4TR\02-UKR_cleaned_data\02-UKR_cleaned_data",
        help="Path to the folder containing TIF images",
        dest="tif_folder_path",
    )
    parser.add_argument(
        "--output-dir",
        "-od",
        type=str,
        default="outputs/tif_tile_inference_512",
        help="Output directory for results",
        dest="output_dir",
    )
    parser.add_argument(
        "--clip-min-patch-size",
        "-cmps",
        type=int,
        default=100,  # Restored to original value for 512x512 tiles
        help="Minimum patch size for CLIP",
        dest="clip_min_patch_size",
    )
    parser.add_argument(
        "--clip-img-padding",
        "-cip",
        type=int,
        default=10,  # Restored to original value for 512x512 tiles
        help="Padding of patch for CLIP",
        dest="clip_img_padding",
    )
    parser.add_argument(
        "--dino-path",
        "-dp",
        type=str,
        required=True,
        help="Path to the DINO model",
        dest="dino_path",
    )
    parser.add_argument(
        "--dino-config",
        "-dc",
        type=str,
        required=True,
        help="Path to the DINO config file",
        dest="dino_config",
    )
    parser.add_argument(
        "--dino-threshold",
        "-dt",
        type=float,
        default=0.15,
        help="Threshold for DINO bounding box prediction",
        dest="dino_threshold",
    )
    parser.add_argument(
        "--sam-path",
        "-sp",
        type=str,
        required=True,
        help="Path to the SAM model",
        dest="sam_path",
    )
    parser.add_argument(
        "--save-annotations",
        "-sa",
        action="store_true",
        help="Save annotations",
        dest="save_annotations",
    )
    parser.add_argument(
        "--tile-size",
        "-ts",
        type=int,
        default=512,
        help="Size of tiles for processing (default: 512)",
        dest="tile_size",
    )
    parser.add_argument(
        "--overlap",
        "-ov",
        type=int,
        default=64,
        help="Overlap between tiles in pixels (default: 64)",
        dest="overlap",
    )
    return parser.parse_args()


def main():
    args = get_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print(f"Using device: {device_str}")
    print(f"TIF folder: {args.tif_folder_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Tile overlap: {args.overlap}px")

    # Update global tile size if provided
    global TILE_SIZE
    TILE_SIZE = args.tile_size

    # Load models
    print("Loading DINO model...")
    dino_model, criterion, postprocessors = load_dino_model(
        args.dino_config, args.dino_path, device=device_str
    )

    print("Loading SAM model...")
    sam_model = sam_model_registry["default"](checkpoint=args.sam_path).to(device)
    sam_predictor = SamPredictor(sam_model)
    
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device_str)
    clip_text = clip.tokenize(CONTRASTIVE_PROMPTS).to(device_str)

    # Get list of TIF files
    print("Finding TIF files...")
    inference_dataset = TIFInferenceDataset(args.tif_folder_path)
    tif_files = [inference_dataset[i] for i in range(len(inference_dataset))]

    # Run tile-based inference with metrics
    print(f"Starting {args.tile_size}x{args.tile_size} tile-based inference with metrics...")
    results = ubdd_plusplus_tile_inference(
        dino_model,
        postprocessors,
        sam_predictor,
        clip_text,
        clip_model,
        clip_preprocess,
        args.clip_min_patch_size,
        args.clip_img_padding,
        args.dino_threshold,
        args.save_annotations,
        tif_files,
        device_str,
        args.output_dir,
        args.overlap,  # Pass overlap parameter
    )


if __name__ == "__main__":
    main()

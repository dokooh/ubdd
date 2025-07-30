"""
U-BDD++ Evaluation with pre-trained CLIP - Modified for UKR TIF tile-based inference
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

from segment_anything import SamPredictor, sam_model_registry
import clip

# ...existing imports...
from models.clipmlp.clipmlp import clip_prediction_ensemble, CONTRASTIVE_PROMPTS
from models.dino.util.slconfig import SLConfig
from models.dino.models.registry import MODULE_BUILD_FUNCS
from models.dino.util import box_ops
from utils.filters import preliminary_filter
from utils.utils import pixel_f1_iou

# Constants
TILE_SIZE = 1024
DINO_TEXT_PROMPT = "building"
DAMAGE_DICT_BGR = [[0, 0, 0], [70, 172, 0], [0, 140, 253]]


class TIFTileDataset(Dataset):
    """Dataset class for TIF image tile-based inference"""
    
    def __init__(self, tif_file_path, tile_size=TILE_SIZE, overlap=0):
        self.tif_file_path = tif_file_path
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Load the full TIF image
        try:
            self.full_image = Image.open(tif_file_path).convert('RGB')
            self.original_width, self.original_height = self.full_image.size
            print(f"Loaded TIF image: {self.original_width}x{self.original_height}")
        except Exception as e:
            raise ValueError(f"Error loading TIF file {tif_file_path}: {e}")
        
        # Calculate tile grid
        self.tiles_x = math.ceil(self.original_width / (tile_size - overlap))
        self.tiles_y = math.ceil(self.original_height / (tile_size - overlap))
        self.total_tiles = self.tiles_x * self.tiles_y
        
        print(f"Will create {self.tiles_x}x{self.tiles_y} = {self.total_tiles} tiles")
        
        # Create tile coordinates
        self.tile_coords = []
        for y in range(self.tiles_y):
            for x in range(self.tiles_x):
                # Calculate tile boundaries
                start_x = x * (tile_size - overlap)
                start_y = y * (tile_size - overlap)
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
        
        # DINO transform
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Transform for original image (without normalization)
        self.to_tensor = T.ToTensor()
    
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
        
        # Pad tile to target size if needed
        if tile_image.size != (self.tile_size, self.tile_size):
            # Create a new image with black padding
            padded_image = Image.new('RGB', (self.tile_size, self.tile_size), color='black')
            padded_image.paste(tile_image, (0, 0))
            tile_image = padded_image
        
        # Apply transforms
        pre_image = self.normalize(tile_image)
        pre_image_original = self.to_tensor(tile_image)
        
        return {
            'pre_image': pre_image,
            'pre_image_original': pre_image_original,
            'post_image_original': pre_image_original,
            'tile_info': tile_info,
            'tile_idx': idx,
            'file_name': os.path.basename(self.tif_file_path)
        }
    
    def get_image_info(self):
        """Return information about the original image"""
        return {
            'width': self.original_width,
            'height': self.original_height,
            'tiles_x': self.tiles_x,
            'tiles_y': self.tiles_y,
            'total_tiles': self.total_tiles,
            'tile_size': self.tile_size
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


def merge_tile_predictions(tile_results, image_info, output_path_base):
    """Merge individual tile predictions into a single full-size prediction"""
    
    # Initialize full-size prediction arrays
    full_width = image_info['width']
    full_height = image_info['height']
    
    # Create full-size prediction mask
    full_pred_mask = np.zeros((full_height, full_width), dtype=np.uint8)
    
    # Process each tile result
    for tile_result in tile_results:
        tile_info = tile_result['tile_info']
        pred_mask = tile_result['pred_mask']
        
        # Extract the valid region of the tile (excluding padding)
        valid_height = min(tile_info['height'], pred_mask.shape[0])
        valid_width = min(tile_info['width'], pred_mask.shape[1])
        
        # Place tile prediction in the full image
        full_pred_mask[
            tile_info['start_y']:tile_info['start_y'] + valid_height,
            tile_info['start_x']:tile_info['start_x'] + valid_width
        ] = pred_mask[:valid_height, :valid_width]
    
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
    """Process a single tile and return prediction results"""
    
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
    
    # Convert boxes to xyxy format
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
        
        # CLIP Prediction for each bounding box
        predictions = []
        for bbox in boxes.tolist():
            # Crop out the image given bboxes
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            
            # Apply the image buffer
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
    
    return {
        'pred_mask': pred_mask.cpu().numpy(),
        'num_detections': len(boxes),
        'num_damaged': (pred_mask.cpu().numpy() == 2).sum(),
        'num_undamaged': (pred_mask.cpu().numpy() == 1).sum(),
        'boxes': boxes.tolist(),
        'scores': logits.tolist(),
        'predictions': predictions,
        'tile_info': batch['tile_info']
    }


# Modified inference function for tile-based processing
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
    output_dir
):
    """Run tile-based inference on TIF images and save results"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for tif_file in tqdm(tif_files, desc="Processing TIF files"):
        print(f"\nProcessing: {os.path.basename(tif_file)}")
        
        # Create tile dataset for this TIF file
        tile_dataset = TIFTileDataset(tif_file, tile_size=TILE_SIZE)
        tile_dataloader = DataLoader(tile_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        image_info = tile_dataset.get_image_info()
        tile_results = []
        
        # Process each tile
        with tqdm(tile_dataloader, desc="Processing tiles", leave=False) as tile_pbar:
            for batch in tile_pbar:
                # Extract single item from batch
                single_batch = {k: v[0] if isinstance(v, (list, tuple)) else v[0] 
                              for k, v in batch.items() if k != 'tile_info'}
                single_batch['tile_info'] = batch['tile_info'][0]
                
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
        
        # Merge tiles into full image
        base_name = os.path.splitext(os.path.basename(tif_file))[0]
        output_path_base = os.path.join(output_dir, base_name)
        
        print(f"Merging {len(tile_results)} tiles...")
        full_pred_mask, color_mask = merge_tile_predictions(
            tile_results, image_info, output_path_base
        )
        
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
        
        # Store results
        result = {
            'file_name': os.path.basename(tif_file),
            'image_size': f"{image_info['width']}x{image_info['height']}",
            'num_tiles': len(tile_results),
            'total_detections': total_detections,
            'total_damaged_pixels': int(total_damaged),
            'total_undamaged_pixels': int(total_undamaged),
            'damage_percentage': float(total_damaged) / (image_info['width'] * image_info['height']) * 100,
            'tile_results': tile_results
        }
        all_results.append(result)
        
        print(f"Completed {os.path.basename(tif_file)}: "
              f"{total_detections} detections, "
              f"{total_damaged} damaged pixels, "
              f"{total_undamaged} undamaged pixels")
    
    # Save summary results
    summary_path = os.path.join(output_dir, "tile_inference_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("TIF Tile-based Inference Summary\n")
        f.write("================================\n\n")
        for result in all_results:
            f.write(f"File: {result['file_name']}\n")
            f.write(f"  Image size: {result['image_size']}\n")
            f.write(f"  Number of tiles: {result['num_tiles']}\n")
            f.write(f"  Total detections: {result['total_detections']}\n")
            f.write(f"  Damaged pixels: {result['total_damaged_pixels']}\n")
            f.write(f"  Undamaged pixels: {result['total_undamaged_pixels']}\n")
            f.write(f"  Damage percentage: {result['damage_percentage']:.2f}%\n\n")
    
    print(f"\nTile-based inference completed! Results saved to: {output_dir}")
    print(f"Processed {len(all_results)} TIF images")
    return all_results


def get_args():
    parser = argparse.ArgumentParser(description="TIF Image Tile-based Inference with U-BDD++")
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
        default="outputs/tif_tile_inference",
        help="Output directory for results",
        dest="output_dir",
    )
    parser.add_argument(
        "--clip-min-patch-size",
        "-cmps",
        type=int,
        default=100,
        help="Minimum patch size for CLIP",
        dest="clip_min_patch_size",
    )
    parser.add_argument(
        "--clip-img-padding",
        "-cip",
        type=int,
        default=10,
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
        default=1024,
        help="Size of tiles for processing",
        dest="tile_size",
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

    # Run tile-based inference
    print("Starting tile-based inference...")
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
    )


if __name__ == "__main__":
    main()

"""
U-BDD++ Evaluation with pre-trained CLIP - Modified for UKR TIF inference
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
IMAGE_WIDTH = 1024
DINO_TEXT_PROMPT = "building"
DAMAGE_DICT_BGR = [[0, 0, 0], [70, 172, 0], [0, 140, 253]]


class TIFInferenceDataset(Dataset):
    """Dataset class for TIF image inference"""
    
    def __init__(self, tif_folder_path, dino_transform=True):
        self.tif_folder_path = tif_folder_path
        self.dino_transform = dino_transform
        
        # Find all TIF files in the folder
        self.tif_files = glob.glob(os.path.join(tif_folder_path, "*.tif")) + \
                        glob.glob(os.path.join(tif_folder_path, "*.tiff"))
        
        if len(self.tif_files) == 0:
            raise ValueError(f"No TIF files found in {tif_folder_path}")
        
        print(f"Found {len(self.tif_files)} TIF files for inference")
        
        # DINO transform
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Transform for original image (without normalization)
        self.to_tensor = T.ToTensor()
    
    def __len__(self):
        return len(self.tif_files)
    
    def __getitem__(self, idx):
        tif_path = self.tif_files[idx]
        
        # Load TIF image
        try:
            image = Image.open(tif_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (1024, 1024), color='black')
        
        # Resize to target size
        image = image.resize((IMAGE_WIDTH, IMAGE_WIDTH), Image.Resampling.LANCZOS)
        
        # Apply transforms
        if self.dino_transform:
            pre_image = self.normalize(image)
        else:
            pre_image = self.to_tensor(image)
        
        pre_image_original = self.to_tensor(image)
        
        return {
            'pre_image': pre_image,
            'pre_image_original': pre_image_original,
            'post_image_original': pre_image_original,  # Using same image for both pre/post
            'pre_file_name': tif_path,
            'file_name': os.path.basename(tif_path)
        }


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


# Modified inference function for TIF images
def ubdd_plusplus_inference(
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
    dataloader,
    device,
    output_dir
):
    """Run inference on TIF images and save results"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    with tqdm(dataloader, total=len(dataloader)) as pbar:
        for batch in pbar:
            file_name = batch["file_name"][0]
            pbar.set_description(f"Processing {file_name}")
            
            # Get DINO predictions
            output = get_dino_output(
                dino_model,
                batch["pre_image"][0],
                dino_threshold,
                dino_postprocessors,
                device=device,
            )
            boxes = output["boxes"].detach().cpu()  # cxcywh
            logits = output["scores"].detach().cpu()
            phrases = output["labels"]
            
            # Convert boxes to xyxy format
            boxes = box_convert(boxes * IMAGE_WIDTH, "cxcywh", "xyxy")
            
            # SAM prediction for all bounding boxes
            source_image = (
                (batch["pre_image_original"][0].permute(1, 2, 0) * 255)
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
                    x2_pad = min(int(x2 + image_buffer_x), IMAGE_WIDTH)
                    y2_pad = min(int(y2 + image_buffer_y), IMAGE_WIDTH)
                    
                    pre_building_patch = T.ToPILImage()(
                        batch["pre_image_original"][0, :, y1_pad:y2_pad, x1_pad:x2_pad]
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
            
            # Save results
            base_name = os.path.splitext(file_name)[0]
            
            # Save prediction mask as colored image
            pred_mask_annotate = pred_mask.cpu().numpy()
            color_mask = np.zeros((IMAGE_WIDTH, IMAGE_WIDTH, 3), dtype=np.uint8)
            for i in range(3):
                color_mask[pred_mask_annotate == i] = DAMAGE_DICT_BGR[i]
            
            color_output_path = os.path.join(output_dir, f"{base_name}_prediction_color.png")
            cv2.imwrite(color_output_path, color_mask)
            
            # Save raw prediction mask
            mask_output_path = os.path.join(output_dir, f"{base_name}_prediction_mask.png")
            cv2.imwrite(mask_output_path, pred_mask_annotate.astype(np.uint8))
            
            # Save original image for reference
            original_output_path = os.path.join(output_dir, f"{base_name}_original.png")
            cv2.imwrite(original_output_path, cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR))
            
            # Store results
            result = {
                'file_name': file_name,
                'num_detections': len(boxes),
                'num_damaged': (pred_mask_annotate == 2).sum(),
                'num_undamaged': (pred_mask_annotate == 1).sum(),
                'boxes': boxes.tolist(),
                'scores': logits.tolist(),
                'predictions': predictions
            }
            results.append(result)
            
            pbar.set_postfix(
                detections=len(boxes),
                damaged=result['num_damaged'],
                undamaged=result['num_undamaged'],
                refresh=False
            )
    
    # Save summary results
    summary_path = os.path.join(output_dir, "inference_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("TIF Inference Summary\n")
        f.write("====================\n\n")
        for result in results:
            f.write(f"File: {result['file_name']}\n")
            f.write(f"  Detections: {result['num_detections']}\n")
            f.write(f"  Damaged pixels: {result['num_damaged']}\n")
            f.write(f"  Undamaged pixels: {result['num_undamaged']}\n\n")
    
    print(f"\nInference completed! Results saved to: {output_dir}")
    print(f"Processed {len(results)} TIF images")
    return results


def get_args():
    parser = argparse.ArgumentParser(description="TIF Image Inference with U-BDD++")
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
        default="outputs/tif_inference",
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
    return parser.parse_args()


def main():
    args = get_args()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    print(f"Using device: {device_str}")
    print(f"TIF folder: {args.tif_folder_path}")
    print(f"Output directory: {args.output_dir}")

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

    # Create dataset and dataloader for TIF images
    print("Creating dataset...")
    tif_dataset = TIFInferenceDataset(args.tif_folder_path, dino_transform=True)
    tif_dataloader = DataLoader(tif_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Run inference
    print("Starting inference...")
    results = ubdd_plusplus_inference(
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
        tif_dataloader,
        device_str,
        args.output_dir,
    )


if __name__ == "__main__":
    main()

"""
Extract ground truth masks from GeoJSON files for UKR disaster assessment
"""
import json
import numpy as np
from shapely.geometry import shape, Polygon
from shapely.affinity import translate
import rasterio
from rasterio.features import rasterize
from PIL import Image
import os
from tqdm import tqdm


def load_geojson_annotations(geojson_path):
    """Load annotations from GeoJSON file"""
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    buildings = []
    for feature in geojson_data['features']:
        geometry = shape(feature['geometry'])
        properties = feature.get('properties', {})
        
        # Extract damage level from properties
        # Assuming the property name for damage is 'damage' or 'damage_level'
        damage = properties.get('damage', properties.get('damage_level', 'no-damage'))
        
        buildings.append({
            'geometry': geometry,
            'damage': damage,
            'properties': properties
        })
    
    return buildings


def create_ground_truth_mask(tif_path, geojson_path):
    """Create ground truth mask from GeoJSON annotations"""
    
    # Open the TIF file to get metadata
    with rasterio.open(tif_path) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs
    
    # Load GeoJSON annotations
    buildings = load_geojson_annotations(geojson_path)
    
    # Create mask array (0: background, 1: undamaged building, 2: damaged building)
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Map damage levels to mask values
    damage_mapping = {
        'no-damage': 1,
        'un-classified': 1,  # Treat unclassified as undamaged
        'minor-damage': 2,
        'major-damage': 2,
        'destroyed': 2,
        'moderate-damage': 2
    }
    
    # Rasterize each building polygon
    for building in buildings:
        geometry = building['geometry']
        damage = building['damage'].lower()
        
        # Get mask value based on damage level
        mask_value = damage_mapping.get(damage, 1)  # Default to undamaged if unknown
        
        # Rasterize the polygon
        try:
            # Create a temporary mask for this building
            building_mask = rasterize(
                [(geometry, mask_value)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            
            # Update the main mask (take maximum to handle overlaps)
            mask = np.maximum(mask, building_mask)
            
        except Exception as e:
            print(f"Warning: Could not rasterize building: {e}")
    
    return mask


def save_ground_truth_mask(mask, output_path):
    """Save ground truth mask as PNG"""
    # Create color mask for visualization
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Apply colors (matching DAMAGE_DICT_BGR but in RGB)
    color_mask[mask == 0] = [0, 0, 0]        # Black for background
    color_mask[mask == 1] = [0, 172, 70]     # Green for undamaged
    color_mask[mask == 2] = [253, 140, 0]    # Orange for damaged
    
    # Save as PNG
    Image.fromarray(color_mask).save(output_path)
    
    # Also save binary mask as numpy array
    np.save(output_path.replace('.png', '.npy'), mask)
    
    return mask


def process_city_data(city_name, data_dir, output_dir):
    """Process ground truth data for a single city"""
    
    # File paths
    pre_disaster_tif = os.path.join(data_dir, f"{city_name}_pre_disaster_ortho.tif")
    post_disaster_tif = os.path.join(data_dir, f"{city_name}_post_disaster_ortho.tif")
    geojson_path = os.path.join(data_dir, f"{city_name}_post_disaster_ortho.geojson")
    
    # Check if files exist
    if not os.path.exists(post_disaster_tif):
        print(f"Warning: Post-disaster TIF not found for {city_name}")
        return None
        
    if not os.path.exists(geojson_path):
        print(f"Warning: GeoJSON not found for {city_name}")
        return None
    
    # Create ground truth mask
    print(f"Creating ground truth mask for {city_name}...")
    gt_mask = create_ground_truth_mask(post_disaster_tif, geojson_path)
    
    # Save ground truth mask
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{city_name}_ground_truth.png")
    save_ground_truth_mask(gt_mask, output_path)
    
    # Calculate statistics
    total_pixels = gt_mask.size
    building_pixels = (gt_mask > 0).sum()
    damaged_pixels = (gt_mask == 2).sum()
    undamaged_pixels = (gt_mask == 1).sum()
    
    stats = {
        'city': city_name,
        'total_pixels': int(total_pixels),
        'building_pixels': int(building_pixels),
        'damaged_pixels': int(damaged_pixels),
        'undamaged_pixels': int(undamaged_pixels),
        'building_ratio': float(building_pixels) / float(total_pixels),
        'damage_ratio': float(damaged_pixels) / float(building_pixels) if building_pixels > 0 else 0.0
    }
    
    print(f"  Total pixels: {stats['total_pixels']:,}")
    print(f"  Building pixels: {stats['building_pixels']:,} ({stats['building_ratio']:.2%})")
    print(f"  Damaged pixels: {stats['damaged_pixels']:,}")
    print(f"  Undamaged pixels: {stats['undamaged_pixels']:,}")
    print(f"  Damage ratio: {stats['damage_ratio']:.2%}")
    
    return stats


def extract_all_ground_truth(data_dir, output_dir):
    """Extract ground truth masks for all cities"""
    
    cities = ['Bucha', 'Chernihiv', 'Kerson', 'Mykolaiv']
    all_stats = []
    
    for city in cities:
        print(f"\nProcessing {city}...")
        stats = process_city_data(city, data_dir, output_dir)
        if stats:
            all_stats.append(stats)
    
    # Save overall statistics
    if all_stats:
        stats_path = os.path.join(output_dir, 'ground_truth_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print("\nOverall Statistics:")
        total_pixels = sum(s['total_pixels'] for s in all_stats)
        total_buildings = sum(s['building_pixels'] for s in all_stats)
        total_damaged = sum(s['damaged_pixels'] for s in all_stats)
        
        print(f"  Total pixels across all cities: {total_pixels:,}")
        print(f"  Total building pixels: {total_buildings:,} ({total_buildings/total_pixels:.2%})")
        print(f"  Total damaged pixels: {total_damaged:,} ({total_damaged/total_buildings:.2%})")
    
    return all_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ground truth masks from GeoJSON files")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing TIF and GeoJSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ground_truth_masks",
        help="Output directory for ground truth masks"
    )
    
    args = parser.parse_args()
    
    # Extract all ground truth masks
    extract_all_ground_truth(args.data_dir, args.output_dir)

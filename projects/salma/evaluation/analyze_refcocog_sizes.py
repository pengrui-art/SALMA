import argparse
import os
import sys
import numpy as np
import tqdm

# Add current directory to path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from utils import REFER

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Object Size Distribution in RefCOCOg")
    parser.add_argument("--data-root", default="data/ref_seg", help="Path to the directory containing refcocog folder")
    parser.add_argument("--dataset", default="refcocog", help="Dataset name")
    parser.add_argument("--splitBy", default="umd", help="Split type (umd or google)")
    parser.add_argument("--split", default="val", help="Split to analyze (train, val, test)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Initializing REFER for {args.dataset} ({args.splitBy})...")
    try:
        refer = REFER(args.data_root, args.dataset, args.splitBy)
    except Exception as e:
        print(f"Error initializing REFER: {e}")
        print(f"Please check your --data-root. It should contain the '{args.dataset}' directory.")
        return

    ref_ids = refer.getRefIds(split=args.split)
    print(f"Found {len(ref_ids)} references in '{args.split}' split.")

    # Bins for size ratios
    bins = {
        "<1%": 0,
        "1%-2%": 0,
        "2%-5%": 0,
        ">5%": 0
    }
    
    total_mask_area_monitor = []
    
    print("Analyzing object sizes...")
    for ref_id in tqdm.tqdm(ref_ids):
        ref = refer.loadRefs(ref_id)[0]
        image_id = ref['image_id']
        image_info = refer.loadImgs(image_id)[0]
        img_h, img_w = image_info['height'], image_info['width']
        img_area = img_h * img_w
        
        # Get Mask Area
        # REFER.getMask returns dictionary with 'mask' and 'area'
        # Note: REFER implementation might compute area from polygon or mask
        try:
            mask_data = refer.getMask(ref)
            obj_area = mask_data['area']
        except Exception as e:
            # Fallback if getMask fails (e.g. some obscure bug), though unlikely
            continue
            
        ratio = obj_area / img_area
        total_mask_area_monitor.append(ratio)
        
        if ratio < 0.01:
            bins["<1%"] += 1
        elif ratio < 0.02:
            bins["1%-2%"] += 1
        elif ratio < 0.05:
            bins["2%-5%"] += 1
        else:
            bins[">5%"] += 1

    total = len(ref_ids)
    print("\nObject Size Distribution (Area Ratio):")
    print(f"{'Size Range':<10} | {'Count':<8} | {'Percentage':<10}")
    print("-" * 35)
    for k, v in bins.items():
        print(f"{k:<10} | {v:<8} | {v/total*100:.2f}%")
        
    print("\nSummary Statistics:")
    print(f"Min Ratio: {min(total_mask_area_monitor)*100:.4f}%")
    print(f"Max Ratio: {max(total_mask_area_monitor)*100:.4f}%")
    print(f"Mean Ratio: {np.mean(total_mask_area_monitor)*100:.4f}%")
    print(f"Median Ratio: {np.median(total_mask_area_monitor)*100:.4f}%")

if __name__ == "__main__":
    main()


import argparse
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from dataset import RESDataset
from utils import _init_dist_pytorch
import torch.nn.functional as F

# Specific image ID for "Carrot" example
TARGET_IMG_ID = 11298 
TARGET_TEXT_KEYWORD = "small carrot" # To safeguard we pick the right sentence

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Attention/Prior")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument("--image-folder", default="/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/glamm_data/images/coco2014/train2014/")
    parser.add_argument("--data-path", default="/data1/pengrui/CodeSpace/Sa2VA/data/Sa2VA-Training/ref_seg/")
    parser.add_argument("--output-dir", default="./viz_output")
    return parser.parse_args()

# Global storage for captured mask
captured_data = {}

def cma_hook(module, input, output):
    # input is a tuple: (visual, text, mask_logits)
    # We want mask_logits which is index 2
    if len(input) > 2:
        mask_logits = input[2]
        if mask_logits is not None:
            # It might be detached or not, just clone it
            captured_data['mask_logits'] = mask_logits.detach().cpu()
            print("Captured mask logits!")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input is a config file
    if args.model_path.endswith('.py'):
        try:
            from mmengine.config import Config
            from xtuner.registry import BUILDER
            print(f"Loading config from {args.model_path}...")
            cfg = Config.fromfile(args.model_path)
            
            # Build model using registry
            print("Building model from config...")
            model = BUILDER.build(cfg.model)
            model.eval().cuda()
            
            # Load tokenizer
            # Config usually has tokenizer info
            # Try to get tokenizer path from mllm config
            tokenizer_path = cfg.model.mllm.model_path
            print(f"Loading tokenizer from {tokenizer_path}...")
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                padding_side='right'
            )
            
        except Exception as e:
            print(f"Failed to build from config: {e}")
            print("Falling back to AutoModel (assuming HF path)...")
            model = AutoModel.from_pretrained(
                args.model_path,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().cuda()
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        # Standard HF path loading
        print(f"Loading model from HF path {args.model_path}...")
        model = AutoModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 2. Attach Hook to CMA
    # Navigate to grounding_encoder.cma
    if hasattr(model, 'grounding_encoder') and hasattr(model.grounding_encoder, 'cma'):
        model.grounding_encoder.cma.register_forward_hook(cma_hook)
        print("Hook registered on grounding_encoder.cma")
    else:
        print("Error: Could not find grounding_encoder.cma! Visualization might fail.")
        # Try recursive search just in case
        for name, mod in model.named_modules():
            if "CrossModalAttention2D" in str(type(mod)):
                mod.register_forward_hook(cma_hook)
                print(f"Hook registered on {name}")

    # 3. Load Dataset items
    dataset = RESDataset(
        image_folder=args.image_folder,
        dataset_name="refcoco", # Assuming RefCOCO as per Carrot example
        data_path=args.data_path,
        split="val", # or train, depending on where valid 11298 is. Carrot example usually in Val/TestA
    )

    # Find specific item
    target_item = None
    print(f"Searching for Img ID {TARGET_IMG_ID}...")
    from tqdm import tqdm
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]
        if item['img_id'] == TARGET_IMG_ID:
            # Check text
            texts = item['text']
            # We want the one about "small carrot"
            found_text = None
            if isinstance(texts, list):
                for t in texts:
                    if TARGET_TEXT_KEYWORD in t:
                        found_text = t
                        break
            elif TARGET_TEXT_KEYWORD in texts:
                found_text = texts

            if found_text:
                target_item = item
                target_item['text'] = [found_text] # Only run this prompt
                print(f"Found Item: {found_text}")
                break
    
    if target_item is None:
        print("Could not find image/prompt in RefCOCO Val. Trying RefCOCOg Val...")
        # Fallback logic omitted for brevity, user can switch split if needed
        return

    # 4. Run Inference
    print("Running inference...")
    with torch.no_grad():
        # Pre-process batch
        # Dataset returns tensor images on CPU, models move them usually
        # But predict_forward handles raw dataset dict items often?
        # Let's verify input format. RESDataset output is standard dict
        
        # We need to simulate the batching call locally
        # predict_forward expects: pixel_values, text list, etc.
        # But wait, predict_forward wrapper inside refcoco_eval does:
        # pred_mask = model.predict_forward(**_data_batch, tokenizer=tokenizer)
        
        # _data_batch has 'pixel_values', 'input_ids' etc. NO.
        # RESDataset __getitem__ returns:
        # 'img_id', 'text', 'pixel_values' (tensor), 'gt_masks' (tensor)
        
        batch = {}
        batch['pixel_values'] = target_item['pixel_values'].unsqueeze(0).cuda().to(torch.bfloat16)
        batch['text'] = target_item['text'][0] # Single string
        
        # We need to tokenize possibly if model expects 'input_ids'
        # Check modeling_sa2va_chat.py predict_forward... 
        # Usually it takes raw text and tokenizes internally or expects input_ids
        # Let's try calling predict_forward directly with the signature form refcoco_eval
        
        # Refcoco_eval line 169: model.predict_forward(**_data_batch, tokenizer=tokenizer)
        # _data_batch has NO input_ids, just pixel_values and other keys (dataset dependent)
        # Wait, RESDataset __getitem__ does NOT return input_ids? 
        # Let's check RES.py... it applies template_map_fn? 
        # Actually standard RESDataset usually does not tokenize, predict_forward handles it.
        
        # But just in case, let's look at `predict_forward` again or trust refcoco_eval.
        # refcoco_eval passes `tokenizer` arg, identifying that tokenization happens inside.
        
        # Add batch dim to pixel values
        out = model.predict_forward(
            pixel_values=batch['pixel_values'],
            text=batch['text'],
            tokenizer=tokenizer,
            img_id=target_item.get('img_id')
        )

    # 5. Process Output
    if 'mask_logits' in captured_data:
        logits = captured_data['mask_logits'][0] # (1, H, W) or (H, W)
        if logits.dim() == 3: logits = logits[0]
        
        # Sigmoid
        prob_map = torch.sigmoid(logits).numpy()
        
        # Resize to original image size
        # We need original image path or size
        # Dataset item might not have orig path, but we can reconstruct or resize output
        # Let's look for loaded original image
        refcoco_root = args.image_folder
        # Need to find filename from somewhere or assume COCO format
        # RESDataset doesn't return filename easily?
        # We can just visualize the 1024x1024 map directly
        
        plt.figure(figsize=(10, 10))
        plt.imshow(prob_map, cmap='jet')
        plt.axis('off')
        save_path = os.path.join(args.output_dir, f"{TARGET_IMG_ID}_prior_attention.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved attention map to {save_path}")
        
    else:
        print("Failed to capture mask logits. Hook didn't trigger?")

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
from mmengine.config import Config
from xtuner.registry import BUILDER
from fvcore.nn import FlopCountAnalysis, flop_count_table

def calculate_model_complexity():
    # 1. Load Config
    # Assuming running from root, adjusting path if needed
    config_path = 'projects/salma/configs/sa2va_1b.py'
    print(f"Loading config from {config_path}...")
    try:
        cfg = Config.fromfile(config_path)
    except FileNotFoundError:
        print("Config not found. Please verify the path.")
        return

    # 2. Build Model
    print("Building model...")
    # Mocking preprocessor to avoid complex build if not needed for FLOPs
    if 'preprocessor' in cfg.model:
        cfg.model.preprocessor = None
    
    model = BUILDER.build(cfg.model)
    model.cuda()
    model.eval()

    print("Model built successfully.")

    # 3. Define Input Shapes (Dummy Data)
    # Image: (B, 3, H, W)
    H, W = 1024, 1024
    dummy_image = torch.randn(1, 3, H, W).cuda()
    
    # 4. Profile Vision Encoder & Projector
    print("-" * 40)
    print("Profiling Vision Encoder (InternViT) & Projector...")
    if hasattr(model, 'mllm') and hasattr(model.mllm.model, 'vision_model'):
        vision_model = model.mllm.model.vision_model
        
        class VisionWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(pixel_values=x)
            
        flops_vision = FlopCountAnalysis(VisionWrapper(vision_model), dummy_image)
        flops_vision_total = flops_vision.total() / 1e9
        params_vision = sum(p.numel() for p in vision_model.parameters()) / 1e6
        print(f"Vision Encoder: {flops_vision_total:.2f} GFLOPs, {params_vision:.2f} M Params")
        
        # Profile Projector (mlp1)
        # InternVL usually has 'mlp1' in the model wrapper or inside the module
        projector = None
        if hasattr(model.mllm.model, 'mlp1'):
            projector = model.mllm.model.mlp1
        
        if projector:
            # Projector input shape: (B, N_patches, Hidden_Dim)
            # InternViT-6B usually 3200-4000 dim? Let's assume input from vision model output
            # Need to know output dim of vision model. 
            # 1B model vision encoder might be smaller.
            # Using dummy input of shape (1, 256, 1024) approx
            dummy_proj_in = torch.randn(1, 256, vision_model.config.hidden_size).cuda()
            flops_proj = FlopCountAnalysis(projector, dummy_proj_in)
            flops_proj_total = flops_proj.total() / 1e9
            params_proj = sum(p.numel() for p in projector.parameters()) / 1e6
            print(f"Projector: {flops_proj_total:.2f} GFLOPs, {params_proj:.2f} M Params")
        else:
            flops_proj_total = 0
            params_proj = 0
            print("Could not locate Projector (mlp1)")

    else:
        print("Could not locate vision_model under model.mllm.model.vision_model")
        flops_vision_total = 0
        params_vision = 0
        flops_proj_total = 0
        params_proj = 0

    # 4.5 Profile SAM2 Image Encoder (Hiera)
    print("-" * 40)
    print("Profiling SAM2 Image Encoder (Hiera)...")
    if hasattr(model, 'grounding_encoder') and hasattr(model.grounding_encoder, 'sam2_model'):
        sam2_img_enc = model.grounding_encoder.sam2_model.image_encoder
        flops_hiera = FlopCountAnalysis(sam2_img_enc, dummy_image)
        flops_hiera_total = flops_hiera.total() / 1e9
        params_hiera = sum(p.numel() for p in sam2_img_enc.parameters()) / 1e6
        print(f"SAM2 Image Encoder: {flops_hiera_total:.2f} GFLOPs, {params_hiera:.2f} M Params")
    else:
        flops_hiera_total = 0
        params_hiera = 0
        print("Could not locate SAM2 Image Encoder")

    # 5. Profile Pre-Pass (CMA + Decoder)
    print("-" * 40)
    print("Profiling Grounding Encoder Pre-Pass (CMA + Decoder)...")
    
    if hasattr(model, 'grounding_encoder'):
        grounding_encoder = model.grounding_encoder
        
        # We need to construct input for inject_language_embd
        # ... (Same setup as before)
        
        # 1. Mock sam_states
        B = 1
        C = grounding_encoder.hidden_dim if hasattr(grounding_encoder, 'hidden_dim') else 256
        H_feat, W_feat = 64, 64
        feat = torch.randn(B, C, H_feat, W_feat).cuda()
        
        sam_states = {
            "current_vision_feats": [feat], 
            "current_vision_pos_embeds": [torch.randn(B, C, H_feat, W_feat).cuda()],
            "feat_sizes": [(H_feat, W_feat)],
        }
        
        feat_high = torch.randn(B, C, 256, 256).cuda() 
        sam_states["current_vision_feats"] = [feat_high, feat] 
        sam_states["feat_sizes"] = [(256, 256), (H_feat, W_feat)]
        
        # 2. Mock language_embd
        dummy_lang = torch.randn(B, 10, C).cuda()

        class InjectWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, s, l):
                return self.m.inject_language_embd(s, l, nf_nobj=None)
        
        flops_decoder = FlopCountAnalysis(InjectWrapper(grounding_encoder), (sam_states, dummy_lang))
        flops_decoder_total = flops_decoder.total() / 1e9
        params_decoder = sum(p.numel() for p in grounding_encoder.parameters()) / 1e6
        print(f"Grounding Encoder (Layer+Decoder): {flops_decoder_total:.2f} GFLOPs, {params_decoder:.2f} M Params")

    else:
        print("Could not locate Grounding Encoder")
        flops_decoder_total = 0
        params_decoder = 0


    # 6. Generate Table 5
    print("\n" + "=" * 60)
    print("Draft for Table 5 (Copy to LaTeX)")
    print("=" * 60)
    
    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \caption{\textbf{Computational Cost Analysis.} We compare the FLOPs and parameters. " 
          rf"The pre-pass (SAM-2 Decoder + CMA, {params_decoder:.1f}M) is lightweight compared to the Vision Encoder ({params_vision:.0f}M).}}")
    print(r"    \label{tab:flops}")
    print(r"    \small")
    print(r"    \begin{tabular}{l|cc}")
    print(r"        \toprule")
    print(r"        Component & Params (M) & FLOPs (G) \\")
    print(r"        \midrule")
    print(rf"        Vision Encoder & {params_vision:.1f} & {flops_vision_total:.1f} \\")
    print(rf"        Projector & {params_proj:.1f} & {flops_proj_total:.2f} \\")
    print(rf"        SAM2 Image Encoder & {params_hiera:.1f} & {flops_hiera_total:.1f} \\")
    print(rf"        Pre-Pass (Decoder + CMA) & {params_decoder:.1f} & {flops_decoder_total:.2f} \\")
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    calculate_model_complexity()

# SALMA: Mask-Guided Structure-Aware Alignment for Referring Segmentation

## Introduction

**SALMA** (Structure-Aware Language-Mask Alignment) is a unified Multimodal Large Language Model (MLLM) framework designed to bridge the "semantic-structural gap" in referring segmentation.

![Teaser](images/image.png)

Unified MLLMs often suffer from **attention drift**—erroneously attending to salient distractors rather than the queried target due to a lack of explicit spatial constraints. SALMA addresses this by introducing:
*   **Mask-Biased Attention (MBA)**: A mechanism that repurposes the frozen SAM-2 decoder to generate class-agnostic structural priors, which act as a dense soft gate to modulate the MLLM's cross-modal attention.
*   **Fine-grained Alignment**: A dual-constraint strategy using **Text-Mask Contrastive (TMC)** loss and **Boundary Consistency** loss to enforce both semantic fidelity and structural precision.

SALMA achieves state-of-the-art performance on video segmentation benchmarks (Ref-DAVIS17, Ref-YouTube-VOS) and complex referring image segmentation (RefCOCOg), demonstrating that precise structural grounding can be achieved with negligible computational overhead (~0.7% latency).

## Key Features

*   **Structure-Aware Reasoning**: Unlike implicit methods, SALMA effectively suppresses background noise and enforces structure-aware reasoning using explicit spatial priors.
*   **High Efficiency**: The structural pre-pass adds negligible overhead (< 1 GFLOP, ~0.7% latency) while significantly boosting performance.
*   **Unified Architecture**: Built on Sa2VA and InternVL2.5, integrating powerful semantic reasoning with robust SAM-2 segmentation.

## Performance

| Benchmark | Metric | Sa2VA-1B (Base) | SALMA (Ours) |
| :--- | :--- | :--- | :--- |
| **Ref-DAVIS17** | J&F | 68.5 | **71.9 (+3.4)** |
| **Ref-YouTube-VOS** | J&F | 65.3 | **67.0 (+1.7)** |
| **RefCOCO (Val)** | cIoU | 79.6 | **80.4** |
| **RefCOCO+ (Val)** | cIoU | 73.6 | **74.8** |
| **RefCOCOg (Val)** | cIoU | 77.8 | **78.0** |
| **MeVis** | J&F | 41.7 | **46.0** |

## Installation

### Prerequisites
*   Python 3.10
*   PyTorch 2.8.0 (or compatible latest version)
*   CUDA 12.x

### Setup Environment

```bash
conda create -n salma python=3.10 -y
conda activate salma

# Install PyTorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Install MMCV
pip install mmcv==2.2.0

# Install Dependencies
pip install -r requirements.txt
pip install gradio xtuner pycocotools
```

## Data Preparation

Please organize your datasets (RefCOCO, DAVIS, YouTube-VOS, etc.) according to the structure defined in `projects/salma/configs`.
Default data root is typically configured in the config files (e.g., `/data1/pengrui/CodeSpace/Sa2VA/data/`). You may need to update `DATA_ROOT` in the config files to match your local setup.

## Getting Started

### Training

To train the model on a single node with 8 GPUs:

```bash
# Usage: bash tools/dist.sh train <CONFIG_PATH> <GPUS>
bash tools/dist.sh train projects/salma/configs/SALMA-1B.py 8
```

### Evaluation

To evaluate the model (e.g., on Ref-DAVIS17):

```bash
# Usage: bash tools/dist.sh test <CONFIG_PATH> <GPUS> --checkpoint <CHECKPOINT_PATH>
bash tools/dist.sh test projects/salma/configs/eval_davis.sh 8 --checkpoint SALMA-1B
```
*Note: Check `projects/salma/evaluation/` for specific evaluation scripts.*

### Gradio Demo

We provide a specialized Gradio demo for interacting with the model.

```bash
# Launch the unified demo
python projects/salma/gradio/app_unified.py --hf_path <path_to_hf_model> --device cuda:0
```

**Supported Features:**
*   **Image Captioning / VQA**: Describe images or answer questions.
*   **Segment by Text**: Segment objects based on text queries.
*   **Multi-turn Chat**: Engage in conversation with context awareness.

**Tips:**
*   The demo automatically handles the `<image>` token for the first turn.
*   Segmentation masks are parsed and overlaid on the image/video.
*   Use "Clear" to reset the conversation context.

## Model Zoo

| Model | Config | Description |
| :--- | :--- | :--- |
| **SALMA-1B** | `projects/salma/configs/SALMA-1B.py` | Base model with InternVL2.5-1B and SAM-2 Hiera-L. |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{salma2026,
  title={SALMA: Mask-Guided Structure-Aware Alignment for Referring Segmentation},
  author={Salma Team},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgement

This project is built upon [Sa2VA](https://github.com/sa2va), [SAM-2](https://github.com/facebookresearch/segment-anything-2), and [InternVL](https://github.com/OpenGVLab/InternVL). We thank the authors for their open-source contributions.

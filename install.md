## MPR-CMA Environment install

```bash
conda create -n cma python=3.10 -y
conda activate cma
### install pytorch
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
### install mmcv, we use 2.2.0 as default version.
pip install mmcv==2.2.0

pip install -r requirements.txt
pip install gradio xtuner pycocotools
```

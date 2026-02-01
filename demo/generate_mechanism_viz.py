import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
import os

# ================= 配置区域 =================
# 请确保这里指向一张存在的图片，如果没有 teaser.jpg，请替换为你自己的测试图片路径
IMAGE_PATH = (
    r"d:\32071\Downloads\sa2va-MaskCMA-master\sa2va-cma\assets\images\teaser.jpg"
)
OUTPUT_DIR = r"d:\32071\Downloads\sa2va-MaskCMA-master\sa2va-cma\demo\mechanism_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色配置 (Matplotlib colormaps)
BASELINE_CMAP = "jet"  # 传统热力图，看起来比较"弥散"
OURS_COLOR = [79, 134, 247]  # #4F86F7 (你的论文主色调: 蓝色)
# ===========================================


def create_dummy_mask(shape):
    """
    创建一个模拟的 Mask (高斯分布)，实际使用时请替换为你模型输出的真实 Mask
    """
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    # 创建一个位于中心的椭圆 Mask
    mask = np.exp(
        -(
            (x - center_x) ** 2 / (2 * (w / 4) ** 2)
            + (y - center_y) ** 2 / (2 * (h / 4) ** 2)
        )
    )
    return mask


def apply_heatmap(image, heatmap, cmap_name="jet", alpha=0.5):
    """将热力图叠加到图片上"""
    # 归一化热力图
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # 应用色图
    cmap = plt.get_cmap(cmap_name)
    heatmap_colored = cmap(heatmap_norm)[..., :3] * 255
    heatmap_colored = heatmap_colored.astype(np.uint8)

    # 转换为 PIL Image
    heatmap_img = Image.fromarray(heatmap_colored)

    # 调整大小以匹配原图 (以防万一)
    if heatmap_img.size != image.size:
        heatmap_img = heatmap_img.resize(image.size, Image.BILINEAR)

    # 叠加
    # 使用 blend 混合
    overlay = Image.blend(image, heatmap_img, alpha)
    return overlay


def apply_mask_overlay(image, mask, color, alpha=0.6):
    """将二值/软 Mask 以单一颜色叠加到图片上 (MPR-CMA风格)"""
    img_arr = np.array(image)

    # 创建纯色层
    color_layer = np.zeros_like(img_arr)
    color_layer[:] = color

    # 扩展 mask 维度
    if len(mask.shape) == 2:
        mask = mask[:, :, np.newaxis]

    # 仅在 Mask 区域混合颜色
    # mask > 0.1 的区域应用颜色
    mask_indices = mask > 0.1

    # 混合: Original * (1-alpha) + Color * alpha
    # 但只在 mask 区域
    blended = img_arr.copy()
    # 注意：这里需要处理 mask 维度匹配问题
    mask_bool = mask_indices[:, :, 0]

    blended[mask_bool] = (
        img_arr[mask_bool] * (1 - alpha) + color_layer[mask_bool] * alpha
    ).astype(np.uint8)

    return Image.fromarray(blended)


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        # 创建一个假的图片用于演示
        print("Creating a dummy image for demonstration...")
        img_np = np.zeros((500, 500, 3), dtype=np.uint8) + 200  # 灰色背景
        original_img = Image.fromarray(img_np)
    else:
        # 1. 加载图片
        original_img = Image.open(IMAGE_PATH).convert("RGB")
        img_np = np.array(original_img)
        print(f"Loaded image: {img_np.shape}")

    # 2. 生成模拟数据 (实际制作时，请加载你的真实 attention map 和 mask)
    # Baseline: 模拟全局弥散的注意力 (加很多噪声的高斯)
    dummy_mask = create_dummy_mask(img_np.shape)
    noise = np.random.normal(0, 0.2, dummy_mask.shape)

    # 使用 PIL 进行高斯模糊替代 cv2.GaussianBlur
    # 先将 numpy 数组转为 PIL Image
    mask_plus_noise = dummy_mask + noise
    mask_plus_noise = (
        (mask_plus_noise - mask_plus_noise.min())
        / (mask_plus_noise.max() - mask_plus_noise.min())
        * 255
    )
    mask_img = Image.fromarray(mask_plus_noise.astype(np.uint8))
    baseline_attn_img = mask_img.filter(ImageFilter.GaussianBlur(radius=50))

    # 转回 numpy 归一化
    baseline_attn = np.array(baseline_attn_img) / 255.0

    # Ours: 模拟清晰的 Mask Prior (锐利)
    ours_mask = np.where(dummy_mask > 0.6, 1.0, 0.0)  # 二值化，模拟 Mask Prior

    # 3. 生成可视化图

    # (a) Baseline Visualization: Heatmap style
    viz_baseline = apply_heatmap(
        original_img, baseline_attn, cmap_name=BASELINE_CMAP, alpha=0.5
    )
    viz_baseline.save(os.path.join(OUTPUT_DIR, "viz_a_baseline.png"))
    print("Saved Baseline visualization.")

    # (b) MPR-CMA Visualization: Mask Overlay style
    viz_ours = apply_mask_overlay(original_img, ours_mask, OURS_COLOR, alpha=0.5)
    viz_ours.save(os.path.join(OUTPUT_DIR, "viz_b_ours.png"))
    print("Saved MPR-CMA visualization.")

    print(f"\nDone! Check the folder: {OUTPUT_DIR}")
    print(
        "Tip: For the final paper, replace 'create_dummy_mask' with your actual model outputs!"
    )


if __name__ == "__main__":
    main()

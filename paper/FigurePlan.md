### 接下来你需要制作的图片 (Actionable Items)

请根据论文逻辑，去制作以下两张关键图片。做好后保存为 PDF 或 PNG 格式。

1. **Figure 1: Teaser（放在 Introduction 里的首页图）**

   * **文件名建议** : `teaser.png`
   * **内容** :
   * 左图：Sa2VA Baseline 的结果。找一个复杂的视频帧，比如有很多人，它把主要人物和背景里的路人混在一起了（Attention Drift）。
   * 右图：你的结构感知模型（Ours）的结果。清晰地只分割出了主要人物（Mask Biased 生效）。
   * 可以用不同颜色的框或者 Mask 覆盖来对比。
2. **Figure 2: Framework Overview（放在 Methodology 里的核心架构图）**

   * **文件名建议** : `framework_overview.pdf` (矢量图最好)
   * **内容细节** :
   * **Left (Inputs)** : Video Frames → Visual Encoder (InternVL) → Feature Pyramid (Fv**F**v). Text Instruction → LLM → Text Embeddings.
   * **Middle (Mask-Biased Attention - 核心)** :

     * 从 Fv**F**v 拉出一个箭头，经过一个小的 Conv Head，生成  **Mprior**M**p**r**i**or **(Low-res Mask)** 。
     * 画一个 **Sigmoid** 框，输出  **Gate G**G**** 。
     * 画一个 **Cross-Attention** 模块，上面有一个显眼的乘法符号 ⊗**⊗**，展示 G**G** 是如何乘以 Attention Map 的。
   * **Right (Losses)** :

     * 在 Text Embedding 和 Masked Vision Features 之间画个双向箭头 ↔**↔**，标注  **TMC Loss** 。
     * 在 Pred Mask 和 GT Mask 的边缘画个高亮圈，标注  **Boundary Loss** 。
   * 
3. 图片建议内容 (Figure 3 - MBA Module):

   * 这是一个放大的“显微镜”图。
   * 展示注意力矩阵 (Attention Matrix) 的样子。
   * **左边:** 原始 Attention (充满噪点)。
   * **中间:** 你的 Gate (黑白掩码，只有物体是白的)。
   * **右边:** Modulated Attention (干净的物体热力图)。
   * **存放位置:** 建议放在  **3.2 章节文字的中间或顶部** （LaTeX 通常会自动浮动到页面顶部）。

当你准备好图片后，把它们放到 `docs/paper/figures/` 目录下（如果没有这个目录就建一个），然后在 LaTeX 里把 `\fbox` 占位符替换成真实的 `\includegraphics` 即可。

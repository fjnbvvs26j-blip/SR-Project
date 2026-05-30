# MSPA: Multi-Scale Parallel Attention for Image Classification

《高级机器学习理论》课程报告 — 方向三：提出了创新性的算法思路解决实际问题。

## 概述

本项目提出 **MSPA (Multi-Scale Parallel Attention)**，一种四分支并行注意力模块，在通道、空间、多尺度和频域四个维度同时捕获特征依赖关系。MSPA 嵌入 ConvNeXt-Tiny 构建 MSPA-ConvNeXt，在 CIFAR-100 上以仅 257K 参数增量提升分类精度。

系统对比了 13 个模型（8 经典 CNN + 3 注意力 baseline + 1 公平基线 + 1 本文方法），涵盖准确率、效率、消融分析等多维度评估。

## 三步运行

```bash
git clone https://github.com/fjnbvvs26j-blip/SR-Project.git
cd SR-Project && pip install -r requirements.txt
python run_experiments.py --dry_run   # 验证 pipeline（5 epoch）
python run_experiments.py             # 完整实验（13 方法 x 2 数据集）
```

## CIFAR-100 主要结果 (13/13 模型完成)

| 模型 | 类别 | Top-1 Acc | Top-5 Acc | Params (M) |
|------|------|-----------|-----------|------------|
| ViT-B/16 | Transformer | 81.12 | 96.50 | 86.6 |
| ConvNeXt-Tiny | 现代化 CNN | 80.00 | 96.31 | 28.6 |
| SE-ResNet50 | 注意力 CNN | 79.33 | 96.13 | 26.0 |
| Swin-T | Transformer | 77.94 | 95.73 | 28.3 |
| CBAM-ResNet50 | 注意力 CNN | 76.93 | 95.47 | 28.1 |
| ECA-ResNet50 | 注意力 CNN | 66.33 | 89.37 | 25.6 |
| EfficientNet-B2 | 现代化 CNN | 64.78 | 89.29 | 9.1 |
| DenseNet-121 | 经典 CNN | 64.45 | 89.19 | 8.0 |
| ResNet-50 | 经典 CNN | 61.49 | 87.15 | 25.6 |
| MobileNetV3-Large | 轻量 CNN | 60.62 | 87.02 | 5.4 |
| ShuffleNetV2 | 轻量 CNN | 59.83 | 87.52 | 2.3 |
| ConvNeXt-Tiny-FT | 公平基线 | <u>88.42</u> | <u>98.53</u> | 28.6 |
| **MSPA-ConvNeXt** | **本文方法** | **89.04** | **98.41** | **28.9** |

### 增益分解
- $\Delta_{\text{protocol}}$ = ConvNeXt-Tiny-FT − ConvNeXt-Tiny = **+8.42pp** — 扩展训练协议增益
- $\Delta_{\text{MSPA}}$ = MSPA-ConvNeXt − ConvNeXt-Tiny-FT = **+0.62pp** — MSPA 模块纯增益

### 关键发现

- **冻结 BN 问题**：BN-based 模型平均 62.23%，LN-based 模型平均 79.69%，差距 17.46pp。在冻结骨干的迁移学习中，LayerNorm 架构天然优于 BatchNorm。
- **注意力补偿效应**：SE-ResNet50 (79.33%) 比 ResNet-50 (61.49%) 提升 17.84pp，验证了全局感受野的通道注意力可有效补偿 BN 统计量不匹配。补偿梯度：SE (+17.84) > CBAM (+15.44) > ECA (+4.84)，与感受野大小正相关。
- **公平基线设计**：ConvNeXt-Tiny-FT 使用与 MSPA 完全相同的扩展训练协议但不含 MSPA 模块，严格隔离模块贡献。MSPA 的纯增益为 +0.62pp。
- **消融实验**：CA+SA 基座 88.61%，MA 和 FA 联合贡献 +0.19pp（30 epoch）。四个分支覆盖"通道-空间-尺度-频率"互补信息维度。
- **小样本局限**：Flowers-102 上 MSPA-ConvNeXt (93.09%) 相比公平基线 (93.56%) 出现 -0.47pp 退化，揭示了注意力模块对训练数据量的依赖性。
- **逐样本定性分析**：全测试集逐样本对比中，MSPA 修正基线错误 302 例，基线修正 MSPA 错误 235 例，MSPA 净胜出 67 例。MSPA 在跨域混淆（crab/rocket, plain/rocket）和同类细粒度（oak/maple tree）场景中表现优势。

## Flowers-102 跨数据集验证

| 模型 | Top-1 Acc | Top-5 Acc | Macro F1 |
|------|-----------|-----------|----------|
| ConvNeXt-Tiny-FT | **93.56%** | **98.50%** | **0.934** |
| MSPA-ConvNeXt | 93.09% | **98.52%** | 0.928 |
| ConvNeXt-Tiny (冻结) | 86.11% | 95.97% | 0.859 |
| SE-ResNet50 (冻结) | 79.33% | 92.16% | 0.784 |

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/fjnbvvs26j-blip/SR-Project.git
cd SR-Project

# 安装依赖
pip install -r requirements.txt

# 冒烟测试（5 epoch，快速验证 pipeline）
python run_experiments.py --dry_run

# 运行全部实验（13方法 × 2数据集）
python run_experiments.py

# 仅运行特定模型
python run_experiments.py --models convnext_tiny convnext_tiny_ft mspa_convnext

# 消融实验
python run_experiments.py --ablation

# 生成论文图表
python draw_figures.py

# 生成可视化对比图
python visualize_results.py
```

> 首次运行会自动下载 CIFAR-100 (169MB) 和 Flowers-102 数据集。需要 GPU (12GB+ VRAM 推荐)。

## 项目结构

```
SR-Project/
├── run_experiments.py        # 主实验脚本（统一框架）
├── draw_figures.py           # 论文图表生成（9张）
├── visualize_results.py      # 实验对比可视化
├── parse_results.py          # 结果解析工具
├── our_method/
│   ├── mspa.py               # MSPA 模块实现（4分支并行注意力）
│   └── mspa_convnext.py      # MSPA-ConvNeXt 完整模型
├── baselines/
│   ├── se_resnet.py          # SE-ResNet50
│   ├── cbam_resnet.py        # CBAM-ResNet50
│   └── eca_resnet.py         # ECA-ResNet50
├── report/
│   └── 课程报告_v3.md        # 正式报告（含 LaTeX 公式）
├── data/                     # CIFAR-100 + Flowers-102
├── results/
│   ├── tables/               # 实验结果 JSON
│   ├── figures/              # 可视化图表
│   └── checkpoints/          # 模型权重
└── README.md
```

## 对比方法

| 类别 | 方法 | 参数量 |
|------|------|--------|
| 经典 CNN | ResNet-50, DenseNet-121 | 25.6M / 8.0M |
| 轻量 CNN | MobileNetV3-L, ShuffleNetV2 | 5.4M / 2.3M |
| 现代化 CNN | EfficientNet-B2, ConvNeXt-Tiny | 9.1M / 28.6M |
| Transformer | ViT-B/16, Swin-T | 86.6M / 28.3M |
| 注意力基线 | SE-ResNet50, CBAM-ResNet50, ECA-ResNet50 | ~26-28M |
| 公平基线 | ConvNeXt-Tiny-FT | 28.6M |
| **本文方法** | **MSPA-ConvNeXt** | **28.9M (+0.26M)** |

## MSPA 创新点

### 四分支并行注意力
- **CA** (Channel Attention): ECA-style 自适应 1D 卷积
- **SA** (Spatial Attention): 1×1 压缩 + 3×3 空间卷积
- **MA** (Multi-Scale Attention): {3,5,7} 三尺度并行取均值
- **FA** (Frequency Attention): FFT → 逐通道复数权重 → IFFT

### 可学习自适应融合
Softmax(α,β,γ,δ) 加权组合，训练中自动调整各分支贡献。

### 公平对比设计
ConvNeXt-Tiny-FT 使用与 MSPA 完全相同的训练协议（解冻最后 2 stage、60 epoch、CutMix、分层 LR）但不含 MSPA 模块，严格隔离模块贡献。

## 评估指标

- Top-1 / Top-5 Accuracy
- Macro F1 Score
- Per-Superclass Accuracy（CIFAR-100 20 超类）
- 参数量 / 推理时间
- 混淆矩阵 / Grad-CAM 热力图 / t-SNE 特征分布
- 消融实验（4 变体）
- 跨数据集验证（Flowers-102）

## 训练协议

| 参数 | 标准基线 | ConvNeXt-Tiny-FT | MSPA-ConvNeXt |
|------|----------|-------------------|---------------|
| Epochs | 30 | 60 | 60 |
| 骨干 | 冻结 | 解冻最后 2 stage | 解冻最后 2 stage |
| 优化器 | AdamW, lr=1e-3 | AdamW, 分层 LR | AdamW, 分层 LR |
| 调度器 | CosineAnnealing | CosineAnnealingWarmRestarts | CosineAnnealingWarmRestarts |
| 数据增强 | Flip + ColorJitter | + CutMix(p=0.5) | + CutMix(p=0.5) |

## 环境要求

- Python 3.8+
- PyTorch 2.0.1+cu118
- torchvision 0.15.2
- GPU: 12GB+ VRAM

```bash
pip install torch torchvision matplotlib numpy tqdm scikit-learn
```

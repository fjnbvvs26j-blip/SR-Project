#!/usr/bin/env python3
"""
论文图表生成 — 9 张高质量图表。

图1: 整体框架 (架构图, matplotlib patches)
图2: MSPA 模块结构 (四分支融合图)
图3: 频域注意力数据流
图4: Grad-CAM 热力图 (需要实验后模型)
图5: t-SNE 特征分布 (需要实验后特征)
图6: 混淆矩阵热力图 (需要实验后预测)
图7: 输入输出对比 (需要实验后模型)
图8: 训练曲线 (需要实验结果 JSON)
图9: 参数-精度散点图 (需要实验结果 JSON)

依赖:
    pip install matplotlib numpy scikit-learn
    可选: pip install grad-cam (图4)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

FIGURES_DIR = './results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 全局样式
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


# ====================================================================
# 图1: 整体框架图
# ====================================================================

def draw_architecture_diagram(save_path: Optional[str] = None) -> None:
    """
    绘制端到端分类框架: Input → ConvNeXt Backbone → MSPA Modules → Classifier → Output
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Overall Architecture: MSPA-ConvNeXt for Image Classification',
                 fontsize=16, fontweight='bold', pad=20)

    # 颜色方案
    c_input = '#E3F2FD'
    c_backbone = '#BBDEFB'
    c_mspa = '#FFCC80'
    c_head = '#A5D6A7'
    c_output = '#EF9A9A'

    # 框定义: (x, y, w, h, label, color, subtitle)
    boxes = [
        (0.3, 1.5, 2.0, 2.0, 'Input\nImage', c_input, '3×224×224'),
        (3.0, 1.5, 3.5, 2.0, 'ConvNeXt-Tiny\nBackbone', c_backbone, '4 Stages, Frozen(0-1)+Trainable(2-3)'),
        (7.2, 1.2, 3.0, 2.6, 'MSPA\nModules', c_mspa, 'Inserted after Stage 2 & 3'),
        (11.0, 1.5, 2.5, 2.0, 'Classification\nHead', c_head, 'LayerNorm + Linear(768→C)'),
        (14.2, 1.5, 1.5, 2.0, 'Output\nClass', c_output, 'Top-1 Prediction'),
    ]

    for x, y, w, h, label, color, subtitle in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                               facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        # 主标签
        ax.text(x + w / 2, y + h / 2 + 0.25, label, ha='center', va='center',
                fontsize=10, fontweight='bold')
        # 副标题
        ax.text(x + w / 2, y + h / 2 - 0.5, subtitle, ha='center', va='center',
                fontsize=7, style='italic', color='#555')

    # 箭头
    arrows = [(2.3, 2.5, 3.0, 2.5), (6.5, 2.5, 7.2, 2.5),
              (10.2, 2.5, 11.0, 2.5), (13.5, 2.5, 14.2, 2.5)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))

    # MSPA 内部标注
    ax.text(8.7, 3.35, 'CA | SA | MA | FA', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#E65100')
    ax.text(8.7, 2.95, 'α·CA + β·SA + γ·MA + δ·FA', ha='center', va='center',
            fontsize=7, color='#E65100')

    # 图例: 冻结 vs 可训练
    ax.text(0.3, 0.5, '■ Frozen ■ Trainable', fontsize=9, color='#333')

    path = save_path or os.path.join(FIGURES_DIR, 'fig1_architecture.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 1: {path}')


# ====================================================================
# 图2: MSPA 模块结构
# ====================================================================

def draw_mspa_module(save_path: Optional[str] = None) -> None:
    """
    绘制 MSPA 四分支并行注意力模块结构图 — CVPR 风格，无文字重叠。
    """
    fig, ax = plt.subplots(1, 1, figsize=(22, 10.5))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13.5)
    ax.axis('off')

    # ---- Color palette ----
    C = {
        'ca_bg': '#FFF3E0', 'ca_edge': '#E65100', 'ca_text': '#BF360C',
        'sa_bg': '#E8F5E9', 'sa_edge': '#2E7D32', 'sa_text': '#1B5E20',
        'ma_bg': '#E3F2FD', 'ma_edge': '#1565C0', 'ma_text': '#0D47A1',
        'fa_bg': '#FCE4EC', 'fa_edge': '#C62828', 'fa_text': '#B71C1C',
        'input_bg': '#F5F5F5', 'input_edge': '#37474F',
        'fusion_bg': '#FFF9C4', 'fusion_edge': '#F57F17',
        'out_bg': '#E0F2F1', 'out_edge': '#00695C',
        'arrow': '#546E7A', 'residual': '#D84315',
    }

    def _rr(x, y, w, h, c, ec, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1',
                     facecolor=c, edgecolor=ec, linewidth=lw, zorder=2))

    def _arr(x1, y1, x2, y2, color='#546E7A', lw=0.9):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                     connectionstyle='arc3,rad=0'))

    # ============================================================
    # TITLE
    # ============================================================
    ax.text(11, 13.0, 'MSPA Module: Multi-Scale Parallel Attention',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # ============================================================
    # INPUT ROW
    # ============================================================
    ix, iy, iw, ih = 3.5, 11.0, 15.0, 1.0
    _rr(ix, iy, iw, ih, C['input_bg'], C['input_edge'], lw=1.8)
    ax.text(ix + iw/2, iy + ih/2, 'Input Feature  X  [ B, C, H, W ]',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # ============================================================
    # FOUR BRANCHES
    # ============================================================
    branch_x = [1.0, 6.0, 11.0, 16.0]
    branch_w = 4.0
    branch_y0 = 5.0
    branch_h = 5.5

    branch_info = [
        {
            'key': 'ca', 'x': branch_x[0],
            'title': 'Channel Attention (CA)',
            'subtitle': 'k = | log2(C)/2 + 1/2 |_odd',
            'ops': ['Global Average Pool', 'Conv1D(k)  [1D conv along C]', 'Sigmoid'],
            'out_shape': 'Output:  [B, C, 1, 1]',
        },
        {
            'key': 'sa', 'x': branch_x[1],
            'title': 'Spatial Attention (SA)',
            'subtitle': '1x1 squeeze + 3x3 spatial conv',
            'ops': ['1x1 Conv:  C  ->  C/r', '3x3 Conv:  C/r  ->  1', 'Sigmoid'],
            'out_shape': 'Output:  [B, 1, H, W]',
        },
        {
            'key': 'ma', 'x': branch_x[2],
            'title': 'Multi-Scale Attention (MA)',
            'subtitle': '{3, 5, 7} three parallel convs -> mean',
            'ops': ['3x3 Conv -> 1ch', '5x5 Conv -> 1ch', '7x7 Conv -> 1ch',
                    'Mean  +  Sigmoid'],
            'out_shape': 'Output:  [B, 1, H, W]',
        },
        {
            'key': 'fa', 'x': branch_x[3],
            'title': 'Frequency Attention (FA)',
            'subtitle': 'FFT  ->  complex weight  ->  IFFT',
            'ops': ['rFFT2D -> Complex', 'Per-ch. Complex Weight', 'irFFT2D -> Real',
                    'Sigmoid'],
            'out_shape': 'Output:  [B, C, H, W]',
        },
    ]

    for bi in branch_info:
        k = bi['key']
        bx = bi['x']

        # Branch background
        _rr(bx, branch_y0, branch_w, branch_h, C[f'{k}_bg'], C[f'{k}_edge'], lw=1.8)

        # Branch title (bold, inside top of box)
        ty = branch_y0 + branch_h - 0.5
        ax.text(bx + branch_w/2, ty, bi['title'],
                ha='center', va='center', fontsize=9.5, fontweight='bold',
                color=C[f'{k}_text'])

        # Branch subtitle (italic, below title)
        sy = ty - 0.55
        ax.text(bx + branch_w/2, sy, bi['subtitle'],
                ha='center', va='center', fontsize=7.5, color='#616161',
                style='italic')

        # Operation boxes — stacked vertically
        op_w = branch_w - 0.5
        op_h = 0.62
        op_gap = 0.18
        op_start_y = sy - 0.65

        for j, op_text in enumerate(bi['ops']):
            oy = op_start_y - j * (op_h + op_gap)
            # White operation box with tinted border
            ax.add_patch(FancyBboxPatch((bx + 0.25, oy), op_w, op_h,
                         boxstyle='round,pad=0.06',
                         facecolor='white', edgecolor=C[f'{k}_edge'],
                         linewidth=0.7, alpha=0.85, zorder=3))
            ax.text(bx + branch_w/2, oy + op_h/2, op_text,
                    ha='center', va='center', fontsize=7.5, zorder=4)

        # Output shape label (below all ops, inside branch bottom)
        last_op_y = op_start_y - (len(bi['ops']) - 1) * (op_h + op_gap)
        os_y = last_op_y - 0.65
        ax.text(bx + branch_w/2, os_y, bi['out_shape'],
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color=C[f'{k}_text'],
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=C[f'{k}_edge'], lw=0.6, alpha=0.7))

    # ============================================================
    # INPUT -> BRANCH CONNECTIONS (diverging)
    # ============================================================
    for bi in branch_info:
        _arr(ix + iw/2, iy, bi['x'] + branch_w/2, branch_y0 + branch_h + 0.06,
             color=C['arrow'], lw=0.9)

    # ============================================================
    # FUSION LAYER
    # ============================================================
    fx, fy, fw, fh = 2.0, 2.5, 18.0, 1.0
    _rr(fx, fy, fw, fh, C['fusion_bg'], C['fusion_edge'], lw=1.8)
    ax.text(fx + fw/2, fy + fh/2,
            'Adaptive Fusion:    A = w_ca * A_ca  +  w_sa * A_sa  +  w_ma * A_ma  +  w_fa * A_fa\n'
            '                    w_i = softmax(alpha), softmax(beta), softmax(gamma), softmax(delta)',
            ha='center', va='center', fontsize=9.5, fontweight='bold')

    # Branch -> Fusion connections (converging)
    for bi in branch_info:
        _arr(bi['x'] + branch_w/2, branch_y0 - 0.06, fx + fw/2, fy + fh + 0.06,
             color=C['arrow'], lw=0.9)

    # ============================================================
    # OUTPUT LAYER
    # ============================================================
    ox, oy, ow, oh = 5.5, 0.8, 11.0, 1.0
    _rr(ox, oy, ow, oh, C['out_bg'], C['out_edge'], lw=2.0)
    ax.text(ox + ow/2, oy + oh/2, 'Output:   Y = A * X  +  X    (residual connection)',
            ha='center', va='center', fontsize=10.5, fontweight='bold')

    # Fusion -> Output
    _arr(fx + fw/2, fy, ox + ow/2, oy + oh + 0.06, color=C['arrow'], lw=1.1)

    # ============================================================
    # RESIDUAL CONNECTION (right-side curved arrow)
    # ============================================================
    res_x = ix + iw + 0.3
    ax.annotate('', xy=(res_x, oy + oh/2), xytext=(res_x, iy + ih/2),
                arrowprops=dict(arrowstyle='->', color=C['residual'], lw=2.2,
                                connectionstyle='arc3,rad=-0.3'), zorder=1)
    ax.text(res_x + 0.6, (iy + oy)/2 + 1.2, 'Residual\n+ X',
            ha='center', va='center', fontsize=8, fontweight='bold',
            color=C['residual'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C['residual'], lw=0.8, alpha=0.9))

    # ============================================================
    # LEGEND
    # ============================================================
    legends = [
        (C['ca_bg'], C['ca_edge'], 'CA: Channel Attention'),
        (C['sa_bg'], C['sa_edge'], 'SA: Spatial Attention'),
        (C['ma_bg'], C['ma_edge'], 'MA: Multi-Scale Attention'),
        (C['fa_bg'], C['fa_edge'], 'FA: Frequency Attention'),
    ]
    lx, ly = 13.0, 0.1
    for i, (bg, ec, label) in enumerate(legends):
        lxi = lx + (i % 2) * 4.5
        lyi = ly - (i // 2) * 0.5
        ax.add_patch(FancyBboxPatch((lxi, lyi), 0.35, 0.35, boxstyle='round,pad=0.06',
                     facecolor=bg, edgecolor=ec, linewidth=1.0, zorder=4))
        ax.text(lxi + 0.5, lyi + 0.17, label, ha='left', va='center',
                fontsize=7.5, zorder=4)

    path = save_path or os.path.join(FIGURES_DIR, 'fig2_mspa_module.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 2: {path}')


# ====================================================================
# 图3: 频域注意力数据流
# ====================================================================

def draw_frequency_attention(save_path: Optional[str] = None) -> None:
    """
    频域注意力分支的数据流: X → FFT → ×W_complex → IFFT → |·| → σ
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('Frequency Attention Branch: FFT → Weighted → IFFT',
                 fontsize=14, fontweight='bold', pad=15)

    steps = [
        (0.5, 'X ∈ ℝ^{B×C×H×W}', '#E3F2FD'),
        (3.0, 'FFT2D\nX̂ ∈ ℂ^{B×C×H×(W/2+1)}', '#BBDEFB'),
        (6.0, 'Channel Weight\nW_real + j·W_imag\n[1, C, 1, 1]', '#FFCC80'),
        (9.5, 'IFFT2D\nŶ ∈ ℝ^{B×C×H×W}', '#C8E6C9'),
        (12.0, 'σ(|Ŷ|)\nAttention Map', '#F8BBD0'),
    ]

    for x, label, color in steps:
        w = 2.0
        rect = FancyBboxPatch((x, 1.2), w, 1.6, boxstyle='round,pad=0.1',
                               facecolor=color, edgecolor='#333', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, 2.0, label, ha='center', va='center',
                fontsize=9, fontweight='bold')

    # 箭头
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 2.0
        x2 = steps[i + 1][0]
        ax.annotate('', xy=(x2, 2.0), xytext=(x1, 2.0),
                     arrowprops=dict(arrowstyle='->', color='#333', lw=2.0))

    # 可学习参数说明
    ax.text(7.0, 3.2, 'Learnable Parameters: W_real, W_imag ∈ ℝ^{1×C×1×1}',
            ha='center', fontsize=10, fontweight='bold', color='#E65100',
            bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.8))
    ax.text(7.0, 0.5, 'Parameter Count: 2×C (e.g., 768 for stage 3)',
            ha='center', fontsize=9, color='#555')

    path = save_path or os.path.join(FIGURES_DIR, 'fig3_frequency_attn.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 3: {path}')


# ====================================================================
# 图4-9: 数据依赖型图表 (需要实验结果)
# ====================================================================

def draw_gradcam_comparison(
    model: object,
    images: torch.Tensor,
    labels: List[int],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Grad-CAM 热力图对比: 有 MSPA vs 无 MSPA (ConvNeXt-T baseline)。
    需要安装: pip install grad-cam
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print('[SKIP] Fig 4: Install pytorch-grad-cam first')
        return

    # TODO: Implement Grad-CAM comparison for 4 typical samples
    # This requires running inference on both models and extracting CAM
    print('[TODO] Fig 4: Grad-CAM comparison (requires trained models)')


def draw_tsne_features(
    features_a: np.ndarray,
    features_b: np.ndarray,
    labels: np.ndarray,
    label_a: str = 'ConvNeXt-T',
    label_b: str = 'MSPA-ConvNeXt',
    save_path: Optional[str] = None,
) -> None:
    """
    t-SNE 特征分布对比。
    """
    from sklearn.manifold import TSNE

    n_samples = min(2000, len(features_a))
    indices = np.random.choice(len(features_a), n_samples, replace=False)
    combined = np.vstack([features_a[indices], features_b[indices]])
    combined_labels = np.hstack([labels[indices], labels[indices]])

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    embedded = tsne.fit_transform(combined)

    mid = n_samples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(embedded[:mid, 0], embedded[:mid, 1], c=combined_labels[:mid],
                cmap='tab20', alpha=0.6, s=5)
    ax1.set_title(f'{label_a}', fontsize=13, fontweight='bold')
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2.scatter(embedded[mid:, 0], embedded[mid:, 1], c=combined_labels[mid:],
                cmap='tab20', alpha=0.6, s=5)
    ax2.set_title(f'{label_b} (Ours)', fontsize=13, fontweight='bold')
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle('t-SNE Feature Distribution Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, 'fig5_tsne.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 5: {path}')


def draw_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
) -> None:
    """混淆矩阵热力图"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues', aspect='auto', vmax=0.3 if normalize else None)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if class_names and len(class_names) <= 20:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)

    path = save_path or os.path.join(FIGURES_DIR, 'fig6_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 6: {path}')


def draw_prediction_examples(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: List[float],
    correct: List[bool],
    save_path: Optional[str] = None,
) -> None:
    """输入输出预测对比: 正确和错误分类的典型样本"""
    n = min(8, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        color = 'green' if correct[i] else 'red'
        ax.set_title(f'True: {true_labels[i]}\nPred: {pred_labels[i]} ({confidences[i]:.1%})',
                     fontsize=8, color=color)
        ax.axis('off')

    for i in range(n, 8):
        axes[i].axis('off')

    fig.suptitle('Prediction Examples (Green=Correct, Red=Error)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, 'fig7_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 7: {path}')


def draw_training_curves_from_results(
    results: Dict,
    highlight_model: str = 'mspa_convnext',
    save_path: Optional[str] = None,
) -> None:
    """从实验结果 JSON 提取并绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for i, (key, r) in enumerate(results.items()):
        history = r.get('training_history', [])
        if not history:
            continue
        name = r.get('model', key)
        epochs = [h['epoch'] for h in history]
        losses = [h['train_loss'] for h in history]
        val_accs = [h['val_acc'] * 100 for h in history]

        is_highlight = highlight_model in name
        alpha = 1.0 if is_highlight else 0.5
        lw = 2.5 if is_highlight else 1.0

        ax1.plot(epochs, losses, color=colors[i], label=name, alpha=alpha, linewidth=lw)
        ax2.plot(epochs, val_accs, color=colors[i], label=name, alpha=alpha, linewidth=lw)

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss'); ax1.legend(fontsize=6); ax1.grid(alpha=0.3)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy'); ax2.legend(fontsize=6); ax2.grid(alpha=0.3)

    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, 'fig8_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 8: {path}')


def draw_params_vs_accuracy_from_results(
    results: Dict,
    save_path: Optional[str] = None,
) -> None:
    """参数-精度散点图, 高亮创新方法"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for key, r in results.items():
        name = r.get('model', key)
        params_m = r.get('params_total', 0) / 1e6
        acc = r['test_top1_acc'] * 100

        is_ours = 'mspa' in name
        color = '#2E7D32' if is_ours else '#1565C0'
        marker = 'D' if is_ours else 'o'
        size = 180 if is_ours else 100
        edge_color = '#FFC107' if is_ours else 'white'

        ax.scatter(params_m, acc, c=color, s=size, marker=marker,
                   edgecolors=edge_color, linewidth=1.5 if is_ours else 0.8,
                   zorder=5 if is_ours else 3)
        ax.annotate(name, (params_m, acc), fontsize=8,
                    textcoords='offset points', xytext=(0, 10), ha='center')

    ax.set_xlabel('Total Parameters (Millions)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs. Model Size', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Pareto 前沿标注
    ax.text(0.98, 0.02, '↑ Better', transform=ax.transAxes,
            fontsize=10, ha='right', va='bottom', color='#555')

    path = save_path or os.path.join(FIGURES_DIR, 'fig9_params_vs_accuracy.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 9: {path}')


# ====================================================================
# 主入口
# ====================================================================

def generate_all_architecture_figures() -> None:
    """生成不依赖实验数据的架构图 (图1-3)"""
    print('Generating architecture figures (Fig 1-3)...')
    draw_architecture_diagram()
    draw_mspa_module()
    draw_frequency_attention()
    print('Done!')


def generate_data_figures(results_path: Optional[str] = None) -> None:
    """生成依赖实验数据的图表 (图4-9)"""
    if results_path is None:
        # 自动加载最新结果
        results_dir = './results/tables'
        files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
        if not files:
            print(f'[ERROR] No results found in {results_dir}')
            return
        results_path = os.path.join(results_dir, files[-1])

    with open(results_path) as f:
        results = json.load(f)

    print(f'Loaded {len(results)} results from {results_path}')
    draw_training_curves_from_results(results)
    draw_params_vs_accuracy_from_results(results)

    # 图4-7 需要额外数据, 此处仅占位
    print('[INFO] Fig 4 (Grad-CAM), Fig 5 (t-SNE), Fig 6 (Confusion), Fig 7 (Predictions)')
    print('       require additional data extraction. Run after experiments complete.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--architecture-only', action='store_true',
                        help='Only generate architecture diagrams (Fig 1-3)')
    parser.add_argument('--data-only', action='store_true',
                        help='Only generate data-dependent figures (Fig 8-9)')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to results JSON for data figures')
    args = parser.parse_args()

    if args.data_only:
        generate_data_figures(args.results)
    elif args.architecture_only:
        generate_all_architecture_figures()
    else:
        generate_all_architecture_figures()
        generate_data_figures(args.results)

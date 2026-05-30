#!/usr/bin/env python3
"""
实验结果可视化。

从 results/tables/*.json 读取实验数据, 生成:
  - 模型对比柱状图 (按数据集分组)
  - 多指标雷达图
  - 参数-精度散点图
  - 训练曲线图
  - 消融实验对比图

用法:
    python visualize_results.py                          # 自动加载最新结果
    python visualize_results.py --results path/to.json   # 指定结果文件
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = './results/tables'
FIGURES_DIR = './results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 模型显示名称映射
MODEL_DISPLAY_NAMES = {
    'resnet50':        'ResNet-50',
    'densenet121':     'DenseNet-121',
    'mobilenetv3':     'MobileNetV3-L',
    'efficientnet_b2': 'EfficientNet-B2',
    'shufflenetv2':    'ShuffleNetV2',
    'convnext_tiny':   'ConvNeXt-T',
    'vit_b16':         'ViT-B/16',
    'swin_t':          'Swin-T',
    'se_resnet50':     'SE-ResNet50',
    'cbam_resnet50':   'CBAM-ResNet50',
    'eca_resnet50':    'ECA-ResNet50',
    'convnext_tiny_ft': 'ConvNeXt-T-FT\n(Fair Baseline)',
    'mspa_convnext':   'MSPA-ConvNeXt\n(Ours)',
}

# 数据集显示名称
DATASET_DISPLAY_NAMES = {
    'cifar100': 'CIFAR-100',
    'flowers102': 'Flowers-102',
}


def load_latest_results() -> Dict:
    """自动加载最新的实验结果 JSON"""
    files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
    if not files:
        raise FileNotFoundError(f'No results found in {RESULTS_DIR}')
    path = os.path.join(RESULTS_DIR, files[-1])
    with open(path) as f:
        return json.load(f)


def load_results_from_path(path: str) -> Dict:
    """从指定路径加载结果"""
    with open(path) as f:
        return json.load(f)


def _display_name(key: str) -> str:
    """将模型/数据集 key 转为显示名称"""
    return MODEL_DISPLAY_NAMES.get(key, DATASET_DISPLAY_NAMES.get(key, key))


def plot_comparison_bar(results: Dict, save_path: Optional[str] = None) -> None:
    """
    模型对比分组柱状图。
    按数据集分组, 每组显示所有模型的 Top-1 Acc 和 Macro F1。
    """
    # 按数据集分组
    by_dataset: Dict[str, Dict[str, dict]] = {}
    for key, r in results.items():
        ds = r.get('dataset', 'unknown')
        model = r.get('model', key)
        by_dataset.setdefault(ds, {})[model] = r

    n_datasets = len(by_dataset)
    fig, axes = plt.subplots(1, n_datasets, figsize=(8 * n_datasets, 6),
                              squeeze=False)
    axes = axes[0]

    for ax, (ds, ds_results) in zip(axes, by_dataset.items()):
        models = list(ds_results.keys())
        names = [_display_name(m) for m in models]
        accs = [ds_results[m]['test_top1_acc'] * 100 for m in models]
        f1s = [ds_results[m]['macro_f1'] * 100 for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width / 2, accs, width, label='Top-1 Acc (%)',
                       color='steelblue', edgecolor='white')
        bars2 = ax.bar(x + width / 2, f1s, width, label='Macro F1 (%)',
                       color='coral', edgecolor='white')

        # 高亮创新方法
        ours_idx = next((i for i, m in enumerate(models) if 'mspa' in m), -1)
        if ours_idx >= 0:
            bars1[ours_idx].set_color('darkgreen')
            bars2[ours_idx].set_color('darkred')

        ax.set_title(_display_name(ds), fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max(max(accs), max(f1s)) * 1.25)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {path}')


def plot_params_vs_accuracy(results: Dict, save_path: Optional[str] = None) -> None:
    """参数-精度散点图: x=总参数量, y=Top-1 Acc"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for key, r in results.items():
        name = _display_name(r.get('model', key))
        x_val = r.get('params_total', 0) / 1e6
        y_val = r['test_top1_acc'] * 100

        color = 'darkgreen' if 'mspa' in key else 'steelblue'
        marker = 's' if 'mspa' in key else 'o'
        size = 120 if 'mspa' in key else 80

        ax.scatter(x_val, y_val, c=color, s=size, marker=marker,
                   edgecolors='white', linewidth=1, zorder=5)
        ax.annotate(name, (x_val, y_val), fontsize=8,
                    textcoords="offset points", xytext=(0, 10), ha='center')

    ax.set_xlabel('Total Parameters (M)', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs. Model Size', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)

    path = save_path or os.path.join(FIGURES_DIR, 'params_vs_accuracy.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[SAVED] {path}')


def plot_training_curves(results: Dict, save_path: Optional[str] = None) -> None:
    """绘制所有模型的训练曲线 (Loss 和 Val Acc vs Epoch)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for i, (key, r) in enumerate(results.items()):
        if 'training_history' not in r or not r['training_history']:
            continue
        history = r['training_history']
        name = _display_name(r.get('model', key))
        epochs = [h['epoch'] for h in history]
        losses = [h['train_loss'] for h in history]
        val_accs = [h['val_acc'] * 100 for h in history]

        alpha = 1.0 if 'mspa' in key else 0.6
        lw = 2.0 if 'mspa' in key else 1.0
        ls = '-' if 'mspa' in key else '-'

        ax1.plot(epochs, losses, color=colors[i], label=name, alpha=alpha, linewidth=lw, linestyle=ls)
        ax2.plot(epochs, val_accs, color=colors[i], label=name, alpha=alpha, linewidth=lw, linestyle=ls)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.legend(fontsize=6, loc='upper right')
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend(fontsize=6, loc='lower right')
    ax2.grid(alpha=0.3)

    fig.suptitle('Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = save_path or os.path.join(FIGURES_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {path}')


def plot_ablation(results: Dict, save_path: Optional[str] = None) -> None:
    """消融实验对比: 4 个 MSPA 变体的指标对比"""
    ablation_keys = ['mspa_full', 'mspa_no_fa', 'mspa_no_ma', 'mspa_base']
    # 找到消融结果 (prefix = 'ablation')
    ablation_results = {k: r for k, r in results.items()
                        if r.get('model', '') in ablation_keys}

    if not ablation_results:
        print('[SKIP] No ablation results found')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels_map = {
        'mspa_full':  'Full MSPA\n(CA+SA+MA+FA)',
        'mspa_no_fa': 'w/o Frequency\n(CA+SA+MA)',
        'mspa_no_ma': 'w/o Multi-Scale\n(CA+SA+FA)',
        'mspa_base':  'Base (CA+SA)\n= CSPA',
    }

    for key, r in ablation_results.items():
        label = labels_map.get(r.get('model', ''), r.get('model', ''))
        acc = r['test_top1_acc'] * 100
        f1 = r['macro_f1'] * 100
        x_pos = list(labels_map.values()).index(label)

        ax.bar(x_pos - 0.15, acc, 0.3, color='steelblue', label='Top-1 Acc' if x_pos == 0 else '')
        ax.bar(x_pos + 0.15, f1, 0.3, color='coral', label='Macro F1' if x_pos == 0 else '')

    ax.set_xticks(range(len(labels_map)))
    ax.set_xticklabels(labels_map.values(), fontsize=10)
    ax.set_ylabel('Score (%)')
    ax.set_title('Ablation Study: MSPA Component Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    path = save_path or os.path.join(FIGURES_DIR, 'ablation.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[SAVED] {path}')


def plot_radar(results: Dict, save_path: Optional[str] = None) -> None:
    """多指标雷达图 (Top-1 Acc, Top-5 Acc, Macro F1, 推理速度的近似: 1/params)"""
    # 过滤到单个数据集
    first_ds = list(results.values())[0].get('dataset', 'cifar100') if results else 'cifar100'
    ds_results = {k: r for k, r in results.items() if r.get('dataset', '') == first_ds}

    if len(ds_results) > 8:
        # 太多模型, 只选取有代表性的
        representative = [
            'resnet50', 'efficientnet_b2', 'convnext_tiny',
            'vit_b16', 'swin_t', 'se_resnet50', 'cbam_resnet50',
            'eca_resnet50', 'mspa_convnext',
        ]
        ds_results = {k: r for k, r in ds_results.items()
                       if r.get('model', k) in representative}

    metrics = ['test_top1_acc', 'test_top5_acc', 'macro_f1']
    metric_labels = ['Top-1 Acc', 'Top-5 Acc', 'Macro F1']

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.tab10(np.linspace(0, 1, len(ds_results)))

    for i, (key, r) in enumerate(ds_results.items()):
        values = [r[m] * 100 for m in metrics]
        values += values[:1]
        name = _display_name(r.get('model', key))
        lw = 2.5 if 'mspa' in key else 1.5
        ax.plot(angles, values, 'o-', color=colors[i], linewidth=lw,
                markersize=5, label=name, alpha=0.85)
        ax.fill(angles, values, alpha=0.03, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_title(f'Multi-Metric Radar — {_display_name(first_ds)}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    path = save_path or os.path.join(FIGURES_DIR, 'radar_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] {path}')


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='实验结果可视化')
    parser.add_argument('--results', type=str, default=None,
                        help='指定结果 JSON 文件路径 (默认: 自动加载最新)')
    parser.add_argument('--all', action='store_true',
                        help='生成所有图表 (默认: 仅柱状图和雷达图)')
    return parser.parse_args()


def main():
    args = parse_args()
    results = load_results_from_path(args.results) if args.results else load_latest_results()
    print(f'[LOADED] {len(results)} experiment results')

    plot_comparison_bar(results)
    plot_radar(results)

    if args.all:
        plot_params_vs_accuracy(results)
        plot_training_curves(results)
        plot_ablation(results)

    print('[DONE] Visualization complete.')


if __name__ == '__main__':
    main()

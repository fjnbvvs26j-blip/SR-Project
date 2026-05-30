#!/usr/bin/env python3
"""
Generate Qualitative Comparison Figure using REAL experimental predictions.
Compares ConvNeXt-Tiny-FT (fair baseline) vs MSPA-ConvNeXt (ours) on CIFAR-100 test samples.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load real predictions
with open(os.path.join(PROJECT_DIR, 'results', 'disagreement_samples.json')) as f:
    data = json.load(f)

samples = data['samples']
classes = data['classes']
stats = data['stats']

# Load CIFAR-100 test images
testset = torchvision.datasets.CIFAR100(
    root=os.path.join(PROJECT_DIR, 'data'), train=False, download=False
)

def create_figure(samples, testset, classes, stats, output_path):
    n = len(samples)

    fig, axes = plt.subplots(n, 4, figsize=(8, 1.6 * n),
                             gridspec_kw={'width_ratios': [0.9, 1.5, 1.5, 0.3]})

    if n == 1:
        axes = axes.reshape(1, -1)

    # Column headers
    headers = ['Input', 'ConvNeXt-Tiny-FT\n(Fair Baseline)', 'MSPA-ConvNeXt\n(Ours)', 'Verdict']
    for j, h in enumerate(headers):
        axes[0, j].set_title(h, fontsize=9, fontweight='bold', pad=6)

    for i, sample in enumerate(samples):
        idx = sample['idx']
        label = sample['label']
        ft_pred = sample['ft_pred']
        mspa_pred = sample['mspa_pred']

        true_name = classes[label]
        ft_name = classes[ft_pred]
        mspa_name = classes[mspa_pred]

        ft_ok = ft_pred == label
        mspa_ok = mspa_pred == label

        # Input image
        img_pil = testset[idx][0]
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_ylabel(true_name, fontsize=8, fontweight='bold',
                              rotation=0, labelpad=25, ha='right', va='center')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # FT prediction
        _draw_pred_cell(axes[i, 1], ft_name, ft_ok, 'ConvNeXt-Tiny-FT')

        # MSPA prediction
        _draw_pred_cell(axes[i, 2], mspa_name, mspa_ok, 'MSPA-ConvNeXt')

        # Verdict
        if mspa_ok and not ft_ok:
            verdict = 'MSPA\nwins'
            v_color = '#2E7D32'
        elif ft_ok and not mspa_ok:
            verdict = 'FT\nwins'
            v_color = '#C62828'
        else:
            verdict = 'Both\nOK'
            v_color = '#757575'

        axes[i, 3].text(0.5, 0.5, verdict, transform=axes[i, 3].transAxes,
                        fontsize=7.5, ha='center', va='center',
                        fontweight='bold', color=v_color)
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        for spine in axes[i, 3].spines.values():
            spine.set_visible(False)

    # Stats annotation
    fig = axes[0, 0].figure
    stats_text = (
        f'Full test set (10,000 samples): '
        f'FT Acc = {stats["ft_acc"]:.2%} | MSPA Acc = {stats["mspa_acc"]:.2%} | '
        f'MSPA wins: {stats["mspa_wins"]} | FT wins: {stats["ft_wins"]} | '
        f'Both correct: {stats["both_correct"]} | Both wrong: {stats["both_wrong"]}'
    )
    fig.text(0.5, 0.005, stats_text, ha='center', fontsize=7.5, fontstyle='italic',
             color='#555555', fontfamily='monospace')

    fig.suptitle('Qualitative Comparison: ConvNeXt-Tiny-FT vs MSPA-ConvNeXt on CIFAR-100',
                 fontsize=11, fontweight='bold', y=0.99)

    plt.tight_layout(pad=1.2)
    plt.subplots_adjust(top=0.92, bottom=0.06)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[SAVED] {output_path}')


def _draw_pred_cell(ax, pred_name, is_correct, model_label):
    color = '#2E7D32' if is_correct else '#C62828'
    bg = '#E8F5E9' if is_correct else '#FFEBEE'
    border = '#2E7D32' if is_correct else '#C62828'
    status = 'CORRECT' if is_correct else 'WRONG'

    ax.text(0.5, 0.55, pred_name, transform=ax.transAxes,
            fontsize=8.5, ha='center', va='center', fontweight='bold',
            color=color, fontfamily='sans-serif')
    ax.text(0.5, 0.30, status, transform=ax.transAxes,
            fontsize=6.5, ha='center', va='center', fontweight='bold',
            color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(1.8)


def main():
    print('=' * 60)
    print('  Qualitative Comparison Figure (REAL predictions)')
    print('=' * 60)
    print(f'Samples: {len(samples)}')
    print(f'Stats: FT={stats["ft_acc"]:.2%}, MSPA={stats["mspa_acc"]:.2%}')
    print(f'  MSPA wins: {stats["mspa_wins"]}, FT wins: {stats["ft_wins"]}')

    output = os.path.join(RESULTS_DIR, 'fig_qualitative_comparison.png')
    create_figure(samples, testset, classes, stats, output)
    print('Done!')


if __name__ == '__main__':
    main()

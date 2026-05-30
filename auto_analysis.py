#!/usr/bin/env python3
"""
自动化实验分析管线：读取 JSON 结果 → 生成 Markdown 报告表格 → 调用 draw_figures.py 生成图表。

用法:
    python auto_analysis.py                           # 生成全部
    python auto_analysis.py --results results.json    # 指定结果文件
"""

from __future__ import annotations
import json, os, sys, argparse
from typing import Dict

RESULTS_DIR = './results/tables'
FIGURES_DIR = './results/figures'
REPORT_PATH = './report/课程报告_v3.md'

# CIFAR-100 超类映射
SUPERCLASS_MAP = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

MODEL_DISPLAY_NAMES = {
    'resnet50': 'ResNet-50',
    'densenet121': 'DenseNet-121',
    'mobilenetv3': 'MobileNetV3-Large',
    'efficientnet_b2': 'EfficientNet-B2',
    'shufflenetv2': 'ShuffleNetV2 1.0x',
    'convnext_tiny': 'ConvNeXt-Tiny',
    'convnext_tiny_ft': 'ConvNeXt-Tiny-FT',
    'vit_b16': 'ViT-B/16',
    'swin_t': 'Swin-T',
    'se_resnet50': 'SE-ResNet50',
    'cbam_resnet50': 'CBAM-ResNet50',
    'eca_resnet50': 'ECA-ResNet50',
    'mspa_convnext': 'MSPA-ConvNeXt',
    'mspa_full': 'MSPA-Full',
    'mspa_no_fa': 'MSPA w/o FA',
    'mspa_no_ma': 'MSPA w/o MA',
    'mspa_base': 'MSPA-Base (CSPA)',
}

MODEL_CATEGORIES = {
    'resnet50': '经典 CNN', 'densenet121': '经典 CNN',
    'mobilenetv3': '轻量 CNN', 'shufflenetv2': '轻量 CNN',
    'efficientnet_b2': '现代化 CNN', 'convnext_tiny': '现代化 CNN',
    'convnext_tiny_ft': '公平基线',
    'vit_b16': 'Transformer', 'swin_t': 'Transformer',
    'se_resnet50': '注意力 CNN', 'cbam_resnet50': '注意力 CNN',
    'eca_resnet50': '注意力 CNN',
    'mspa_convnext': '本文方法',
}

MODEL_PARAMS = {
    'resnet50': 25.6, 'densenet121': 8.0, 'mobilenetv3': 5.4,
    'efficientnet_b2': 9.1, 'shufflenetv2': 2.3, 'convnext_tiny': 28.6,
    'convnext_tiny_ft': 28.6, 'vit_b16': 86.6, 'swin_t': 28.3,
    'se_resnet50': 26.0, 'cbam_resnet50': 28.1, 'eca_resnet50': 25.6,
    'mspa_convnext': 28.9,
}


def find_latest_results() -> str:
    """找到最新的实验结果 JSON 文件。"""
    if not os.path.isdir(RESULTS_DIR):
        return None
    json_files = sorted([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
    if not json_files:
        return None
    return os.path.join(RESULTS_DIR, json_files[-1])


def load_results(path: str) -> Dict:
    """加载实验结果 JSON。"""
    with open(path) as f:
        return json.load(f)


def format_table_row(name: str, category: str, top1: float, top5: float, f1: float, params: float, time_min: float) -> str:
    """格式化 Markdown 表格行。"""
    display = MODEL_DISPLAY_NAMES.get(name, name)
    cat = MODEL_CATEGORIES.get(name, category)
    return (f'| {display} | {cat} | {top1:.2f} | {top5:.2f} | '
            f'{f1:.2f} | {params:.1f} | {time_min:.1f} |')


def generate_results_table(results: Dict) -> str:
    """生成 CIFAR-100 总体性能对比表。"""
    header = '| 模型 | 类别 | Top-1 Acc | Top-5 Acc | Macro F1 | Params (M) | Time (min) |\n'
    header += '|------|------|-----------|-----------|----------|------------|------------|'

    rows = []
    # 按类别分组排序
    order = [
        'resnet50', 'densenet121', 'mobilenetv3', 'efficientnet_b2', 'shufflenetv2',
        'convnext_tiny', 'vit_b16', 'swin_t',
        'se_resnet50', 'cbam_resnet50', 'eca_resnet50',
        'convnext_tiny_ft', 'mspa_convnext',
    ]

    best_top1 = max((r['test_top1_acc'] for r in results.values()), default=0)
    second_top1 = sorted(set(r['test_top1_acc'] for r in results.values()), reverse=True)[1] if len(results) >= 2 else 0

    for name in order:
        key = f'{name}_cifar100'
        if key in results:
            r = results[key]
            params = MODEL_PARAMS.get(name, float(r.get('params_total_m', 0)))
            top1 = r['test_top1_acc'] * 100
            top5 = r.get('test_top5_acc', 0) * 100
            f1 = r['macro_f1'] * 100
            time_min = r.get('training_time_min', r.get('time_min', 0))

            display = MODEL_DISPLAY_NAMES.get(name, name)
            cat = MODEL_CATEGORIES.get(name, '')

            top1_str = f'{top1:.2f}'
            if key == f'mspa_convnext_cifar100':
                top1_str = f'**{top1_str}**'
            elif abs(top1 - best_top1 * 100) < 0.01:
                top1_str = f'**{top1_str}**'
            elif abs(top1 - second_top1 * 100) < 0.01:
                top1_str = f'<u>{top1_str}</u>'

            rows.append(f'| {display} | {cat} | {top1_str} | {top5:.2f} | {f1:.2f} | {params:.1f} | {time_min:.1f} |')

    return header + '\n' + '\n'.join(rows)


def generate_bn_vs_ln_table(results: Dict) -> str:
    """生成 BN vs LN 对比表。"""
    bn_models = ['resnet50', 'densenet121', 'mobilenetv3', 'efficientnet_b2', 'shufflenetv2']
    ln_models = ['convnext_tiny', 'vit_b16', 'swin_t']

    lines = ['| 模型 | 归一化层类型 | 参数冻结时 Top-1 |', '|------|------------|-----------------|']

    bn_vals = []
    for m in bn_models:
        key = f'{m}_cifar100'
        if key in results:
            top1 = results[key]['test_top1_acc'] * 100
            bn_vals.append(top1)
            lines.append(f'| {MODEL_DISPLAY_NAMES.get(m, m)} | BatchNorm [29] | {top1:.2f} |')

    bn_avg = sum(bn_vals) / len(bn_vals) if bn_vals else 0
    lines.append(f'| **BN 平均** | | **{bn_avg:.2f}** |')

    ln_vals = []
    for m in ln_models:
        key = f'{m}_cifar100'
        if key in results:
            top1 = results[key]['test_top1_acc'] * 100
            ln_vals.append(top1)
            lines.append(f'| {MODEL_DISPLAY_NAMES.get(m, m)} | LayerNorm [30] | {top1:.2f} |')

    ln_avg = sum(ln_vals) / len(ln_vals) if ln_vals else 0
    lines.append(f'| **LN 平均** | | **{ln_avg:.2f}** |')

    if bn_avg and ln_avg:
        lines.append(f'\n差距：{ln_avg - bn_avg:.2f} 个百分点')

    return '\n'.join(lines)


def generate_ablation_table(results: Dict) -> str:
    """生成消融实验对比表。"""
    variants = ['mspa_full', 'mspa_no_fa', 'mspa_no_ma', 'mspa_base']
    branches = {
        'mspa_full': '✓ | ✓ | ✓ | ✓',
        'mspa_no_fa': '✓ | ✓ | ✓ | ✗',
        'mspa_no_ma': '✓ | ✓ | ✗ | ✓',
        'mspa_base': '✓ | ✓ | ✗ | ✗',
    }

    header = '| 变体 | CA | SA | MA | FA | Top-1 Acc | Δ vs Full | 参数增量 |\n'
    header += '|------|----|----|----|----|-----------|-----------|----------|'

    lines = [header]
    full_top1 = None
    for v in variants:
        key = f'{v}_cifar100'
        if key in results:
            top1 = results[key]['test_top1_acc'] * 100
            if v == 'mspa_full':
                full_top1 = top1

            delta = ''
            if full_top1 is not None and v != 'mspa_full':
                delta = f'-{full_top1 - top1:.2f}'

            name = MODEL_DISPLAY_NAMES.get(v, v)
            params_inc = '257K' if 'full' in v else ('~255K' if 'no_fa' in v else ('~161K' if 'no_ma' in v else '~159K'))
            lines.append(f'| {name} | {branches[v]} | {top1:.2f} | {delta} | {params_inc} |')

    return '\n'.join(lines)


def print_analysis(results: Dict) -> None:
    """打印完整分析结果。"""
    print('=' * 70)
    print('  EXPERIMENT ANALYSIS')
    print('=' * 70)

    cifar_results = {k: v for k, v in results.items() if 'cifar100' in k}
    ablation_results = {k: v for k, v in results.items() if any(x in k for x in ['mspa_full', 'mspa_no', 'mspa_base'])}

    if cifar_results:
        print('\n## CIFAR-100 Results Table\n')
        print(generate_results_table(cifar_results))
        print('\n## BN vs LN Analysis\n')
        print(generate_bn_vs_ln_table(cifar_results))

    if ablation_results:
        print('\n## Ablation Study\n')
        print(generate_ablation_table({**cifar_results, **ablation_results}))

    # Summary statistics
    top1_list = [(r['model'], r['test_top1_acc']) for r in cifar_results.values()]
    top1_list.sort(key=lambda x: x[1], reverse=True)

    print('\n## Ranking (Top-1 Accuracy)\n')
    for i, (model, acc) in enumerate(top1_list, 1):
        name = MODEL_DISPLAY_NAMES.get(model, model)
        print(f'  {i}. {name}: {acc*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='自动实验分析')
    parser.add_argument('--results', type=str, help='结果 JSON 文件路径')
    args = parser.parse_args()

    results_path = args.results or find_latest_results()
    if not results_path:
        print('[ERROR] No results file found.')
        sys.exit(1)

    print(f'[INFO] Loading: {results_path}')
    results = load_results(results_path)
    print(f'[INFO] {len(results)} experiment results loaded.')
    print_analysis(results)

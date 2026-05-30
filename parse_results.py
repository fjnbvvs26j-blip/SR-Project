#!/usr/bin/env python3
"""解析实验日志，提取所有模型的评估指标，生成结构化 JSON 和汇总表格。"""
import json, os, re, sys
from collections import OrderedDict

def parse_experiment_log(log_path):
    """从实验日志中提取所有模型的评估指标。"""
    if not os.path.exists(log_path):
        print(f'[ERROR] Log file not found: {log_path}')
        return {}

    with open(log_path) as f:
        text = f.read()

    results = OrderedDict()
    # 匹配每个模型的结果: => Best Val: X.XXXX | Test Top-1: X.XXXX | ...
    pattern = r'(\w+)  on  (\w+)  \((\d+) epochs\).*?'
    pattern += r'=> Best Val: ([\d.]+) \| Test Top-1: ([\d.]+) \| Top-5: ([\d.]+) \| Macro F1: ([\d.]+) \| Time: ([\d.]+)min'

    for m in re.finditer(pattern, text, re.DOTALL):
        model, dataset, epochs = m.group(1), m.group(2), int(m.group(3))
        best_val = float(m.group(4))
        test_top1 = float(m.group(5))
        test_top5 = float(m.group(6))
        macro_f1 = float(m.group(7))
        elapsed = float(m.group(8))

        key = f'{model}_{dataset}'
        results[key] = {
            'model': model, 'dataset': dataset, 'epochs': epochs,
            'best_val_acc': best_val,
            'test_top1_acc': test_top1,
            'test_top5_acc': test_top5,
            'macro_f1': macro_f1,
            'training_time_min': elapsed,
        }

    return results


def print_table(results):
    """打印结果汇总表格。"""
    if not results:
        print('No results found.')
        return

    header = f'{"Model":<22} {"Top-1":>8} {"Top-5":>8} {"F1":>8} {"Time(min)":>10}'
    print(header)
    print('-' * len(header))
    best_model, best_acc = '', 0.0
    for v in results.values():
        print(f'{v["model"]:<22} {v["test_top1_acc"]:>8.4f} {v["test_top5_acc"]:>8.4f} '
              f'{v["macro_f1"]:>8.4f} {v["training_time_min"]:>10.1f}')
        if v['test_top1_acc'] > best_acc:
            best_acc = v['test_top1_acc']
            best_model = v['model']
    print(f'\nBest: {best_model} ({best_acc:.4f})')


if __name__ == '__main__':
    log_path = sys.argv[1] if len(sys.argv) > 1 else '/root/experiment_cifar100.log'
    results = parse_experiment_log(log_path)
    if results:
        out_path = '/root/results/tables/parsed_results.json'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[SAVED] {out_path}')
    print_table(results)

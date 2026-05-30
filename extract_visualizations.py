#!/usr/bin/env python3
"""
可视化提取脚本 — 从训练好的模型提取 Grad-CAM / t-SNE / 混淆矩阵。

用法:
    python extract_visualizations.py --model mspa_convnext --dataset cifar100
    python extract_visualizations.py --all

需要先有训练好的模型权重（.pth），或使用 ImageNet 预训练权重做近似。
"""

from __future__ import annotations
import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as sklearn_cm

FIGS_DIR = './results/figures'
os.makedirs(FIGS_DIR, exist_ok=True)

# CIFAR-100 超类
SUPERCLASSES = {
    'aquatic mammals': ['beaver','dolphin','otter','seal','whale'],
    'fish': ['aquarium_fish','flatfish','ray','shark','trout'],
    'flowers': ['orchid','poppy','rose','sunflower','tulip'],
    'food containers': ['bottle','bowl','can','cup','plate'],
    'fruit and vegetables': ['apple','mushroom','orange','pear','sweet_pepper'],
    'household electrical devices': ['clock','keyboard','lamp','telephone','television'],
    'household furniture': ['bed','chair','couch','table','wardrobe'],
    'insects': ['bee','beetle','butterfly','caterpillar','cockroach'],
    'large carnivores': ['bear','leopard','lion','tiger','wolf'],
    'large man-made outdoor things': ['bridge','castle','house','road','skyscraper'],
    'large natural outdoor scenes': ['cloud','forest','mountain','plain','sea'],
    'large omnivores and herbivores': ['camel','cattle','chimpanzee','elephant','kangaroo'],
    'medium mammals': ['fox','porcupine','possum','raccoon','skunk'],
    'non-insect invertebrates': ['crab','lobster','snail','spider','worm'],
    'people': ['baby','boy','girl','man','woman'],
    'reptiles': ['crocodile','dinosaur','lizard','snake','turtle'],
    'small mammals': ['hamster','mouse','rabbit','shrew','squirrel'],
    'trees': ['maple_tree','oak_tree','palm_tree','pine_tree','willow_tree'],
    'vehicles 1': ['bicycle','bus','motorcycle','pickup_truck','train'],
    'vehicles 2': ['lawn_mower','rocket','streetcar','tank','tractor'],
}

# CIFAR-100 class name → superclass
CLASS_TO_SUPERCLASS = {}
for sc, classes in SUPERCLASSES.items():
    for c in classes:
        CLASS_TO_SUPERCLASS[c] = sc


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """构建模型（使用 ImageNet 预训练权重）。"""
    from our_method.mspa import MSPA
    from our_method.mspa_convnext import MSPAConvNeXt

    model_builders = {
        'resnet50': lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'convnext_tiny': lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
        'vit_b16': lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
        'swin_t': lambda: models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
        'mspa_convnext': lambda: MSPAConvNeXt(num_classes=num_classes),
    }
    if model_name in model_builders:
        return model_builders[model_name]()
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device: torch.device,
                     hook_layer: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取特征嵌入、预测和真实标签。"""
    model.eval()
    model.to(device)
    features_list, preds_list, labels_list = [], [], []

    # 注册 hook 提取中间层特征
    features = {}

    def _hook(name):
        def _fn(module, inp, out):
            features[name] = out.detach()
        return _fn

    if hook_layer:
        for name, module in model.named_modules():
            if name == hook_layer:
                module.register_forward_hook(_hook(name))
                break

    for images, labels in loader:
        images = images.to(device)
        output = model(images)
        preds_list.append(output.argmax(1).cpu().numpy())
        labels_list.append(labels.numpy())

        if hook_layer and features:
            f = features[hook_layer]
            if f.dim() == 4:
                f = f.mean([2, 3])  # GAP over spatial dims
            elif f.dim() > 2:
                f = f.reshape(f.size(0), -1)
            features_list.append(f.cpu().numpy())
            features.clear()

    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    feats = np.concatenate(features_list) if features_list else np.array([])
    return feats, preds, labels


def generate_confusion_matrix(model_name: str, preds: np.ndarray, labels: np.ndarray,
                              num_classes: int = 100, dataset: str = 'cifar100') -> str:
    """生成超类级别的混淆矩阵。"""
    # 超类准确率
    cm_full = sklearn_cm(labels, preds, labels=range(num_classes))
    superclass_acc = {}
    for sc_name, class_names in SUPERCLASSES.items():
        indices = []
        cifar100_classes = datasets.CIFAR100(root='./data', download=False).classes
        for cname in class_names:
            if cname in cifar100_classes:
                indices.append(cifar100_classes.index(cname))
        if indices:
            mask = np.isin(labels, indices)
            if mask.sum() > 0:
                sc_preds = preds[mask]
                sc_labels = labels[mask]
                correct = (sc_preds == sc_labels).sum()
                superclass_acc[sc_name] = correct / len(sc_labels) * 100

    # 绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 完整混淆矩阵（取 log 可视化）
    cm_log = np.log1p(cm_full)
    im = ax1.imshow(cm_log, cmap='Blues', aspect='auto')
    ax1.set_title(f'{model_name} - Log Confusion Matrix')
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    plt.colorbar(im, ax=ax1, shrink=0.8)

    # 超类准确率柱状图
    sc_names = list(superclass_acc.keys())
    sc_vals = list(superclass_acc.values())
    colors = ['#2E7D32' if v >= 80 else '#FF9800' if v >= 60 else '#F44336' for v in sc_vals]
    ax2.barh(range(len(sc_names)), sc_vals, color=colors)
    ax2.set_yticks(range(len(sc_names)))
    ax2.set_yticklabels(sc_names, fontsize=7)
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Superclass Accuracy')
    ax2.axvline(x=80, color='green', linestyle='--', alpha=0.5)
    for i, v in enumerate(sc_vals):
        ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=7)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, f'fig6_confusion_{model_name}_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 6: {path}')

    # 返回错误分析报告
    worst = sorted(superclass_acc.items(), key=lambda x: x[1])[:5]
    best = sorted(superclass_acc.items(), key=lambda x: x[1], reverse=True)[:5]
    report = f'\nSuperclass Analysis ({model_name}):\n'
    report += f'  Best: {", ".join(f"{n} ({v:.1f}%)" for n, v in best)}\n'
    report += f'  Worst: {", ".join(f"{n} ({v:.1f}%)" for n, v in worst)}\n'
    return report


def generate_tsne(features_dict: Dict[str, np.ndarray], labels: np.ndarray,
                  dataset: str = 'cifar100', max_samples: int = 3000) -> str:
    """生成 t-SNE 特征分布对比图。"""
    n_models = len(features_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    # 降采样
    idx = np.random.choice(len(labels), min(max_samples, len(labels)), replace=False)

    # 合并所有模型特征做统一 t-SNE（可选）
    for ax, (name, feats) in zip(axes, features_dict.items()):
        feats_sample = feats[idx]
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
        embedded = tsne.fit_transform(feats_sample)

        ax.scatter(embedded[:, 0], embedded[:, 1], c=labels[idx],
                   cmap='tab20', s=3, alpha=0.6)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle('t-SNE Feature Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, f'fig5_tsne_{dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 5: {path}')
    return f't-SNE: {n_models} models, {max_samples} samples'


def generate_gradcam(model: nn.Module, loader: DataLoader, device: torch.device,
                     model_name: str, num_samples: int = 6) -> str:
    """生成简化的 Grad-CAM 热力图（使用最后的卷积层梯度）。"""
    model.eval()
    model.to(device)

    # 找到最后的卷积层
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)

    if last_conv is None:
        return f'[WARN] No Conv2d layer found in {model_name}'

    name, conv_layer = last_conv
    activations = []
    captured_grads = []

    def _save_act(module, inp, out):
        activations.append(out)

    def _save_grad(module, grad_in, grad_out):
        captured_grads.append(grad_out[0])

    hook_act = conv_layer.register_forward_hook(_save_act)
    hook_grad = conv_layer.register_full_backward_hook(_save_grad)

    # 获取几个样本
    images, labels = next(iter(loader))
    images = images[:num_samples].to(device)
    images.requires_grad = True

    output = model(images)
    class_idx = output.argmax(1)

    # 对每个样本计算 Grad-CAM
    one_hot = torch.zeros_like(output)
    for i in range(num_samples):
        one_hot[i, class_idx[i]] = 1

    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=False)

    hook_act.remove()
    hook_grad.remove()

    # 获取激活和梯度
    acts = activations[0][:num_samples]  # [N, C, H, W]
    grads = captured_grads[0][:num_samples]    # [N, C, H, W]

    # Grad-CAM: GAP over gradients → weight * activation → ReLU → upsample
    weights = grads.mean([2, 3], keepdim=True)  # [N, C, 1, 1]
    cam = (weights * acts).sum(1)  # [N, H, W]
    cam = torch.relu(cam)

    # 归一化并绘制
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    plt.suptitle(f'Grad-CAM: {model_name}', fontsize=14, fontweight='bold')

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    for i in range(num_samples):
        # 原图
        img = norm(images[i].detach().cpu().clone())  # de-normalize approximation
        img_np = img.permute(1, 2, 0).detach().numpy()
        img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f'Pred: {class_idx[i].item()}')
        axes[0, i].axis('off')

        # Grad-CAM
        cam_i = cam[i].detach().cpu().numpy()
        cam_i = (cam_i - cam_i.min()) / (cam_i.max() - cam_i.min() + 1e-8)
        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam_i, cmap='jet', alpha=0.5)
        axes[1, i].set_title(f'True: {labels[i].item()}')
        axes[1, i].axis('off')

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, f'fig4_gradcam_{model_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[SAVED] Fig 4: {path}')
    return f'Grad-CAM: {num_samples} samples from {model_name}'


def main():
    parser = argparse.ArgumentParser(description='Extract visualizations from trained models')
    parser.add_argument('--model', type=str, default='convnext_tiny',
                        help='Model to analyze')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'flowers102'])
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations for all key models')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--feature_layer', type=str, default=None,
                        help='Layer name for t-SNE feature extraction (default: before classifier)')
    args = parser.parse_args()

    device = get_device()
    print(f'[DEVICE] {device}')

    # 准备数据
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])

    num_classes = 100 if args.dataset == 'cifar100' else 102
    if args.dataset == 'cifar100':
        test_set = datasets.CIFAR100(root='./data', train=False, transform=test_transform, download=True)
    else:
        test_set = datasets.Flowers102(root='./data', split='test', transform=test_transform, download=True)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    models_to_analyze = [args.model]
    if args.all:
        models_to_analyze = ['resnet50', 'convnext_tiny', 'mspa_convnext', 'se_resnet50']

    reports = []
    for model_name in models_to_analyze:
        print(f'\n{"="*60}')
        print(f'  Processing: {model_name}')
        print(f'{"="*60}')

        model = build_model(model_name, num_classes)
        model.to(device)
        model.eval()

        # 提取特征
        hook = args.feature_layer
        if hook is None and hasattr(model, 'classifier'):
            hook = None  # t-SNE on final features before classifier
        feats, preds, labels = extract_features(model, test_loader, device, hook)

        # 生成混淆矩阵
        if args.dataset == 'cifar100':
            report = generate_confusion_matrix(model_name, preds, labels, num_classes, args.dataset)
            reports.append(report)

        # 生成 Grad-CAM
        report = generate_gradcam(model, test_loader, device, model_name)
        reports.append(report)

    # t-SNE 需要多个模型的特征对比
    if args.all:
        features_dict = {}
        for model_name in models_to_analyze:
            model = build_model(model_name, num_classes)
            feats, _, labels = extract_features(model, test_loader, device)
            if feats.size > 0:
                features_dict[model_name] = feats
        if features_dict:
            report = generate_tsne(features_dict, labels, args.dataset)
            reports.append(report)

    print('\n' + '='*60)
    print('  VISUALIZATION SUMMARY')
    print('='*60)
    for r in reports:
        print(r)


if __name__ == '__main__':
    main()

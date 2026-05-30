#!/usr/bin/env python3
"""
图像分类模型对比实验 — 统一实验框架

支持: 8 经典基线 + 3 注意力基线 + 1 创新方法 (MSPA-ConvNeXt)
数据集: CIFAR-100, Flowers-102
评估: Top-1/5 Acc, Macro P/R/F1, Per-Superclass Acc, 参数量, 推理时间

用法:
    python run_experiments.py                          # 全部实验
    python run_experiments.py --models resnet50 convnext_tiny  # 指定模型
    python run_experiments.py --datasets cifar100       # 指定数据集
    python run_experiments.py --epochs 5 --dry_run      # 快速冒烟测试
    python run_experiments.py --ablation                # 仅消融实验
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import multiprocessing
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    densenet121, DenseNet121_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b2, EfficientNet_B2_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    vit_b_16, ViT_B_16_Weights,
    swin_t, Swin_T_Weights,
)

# --- 项目内模块 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines.se_resnet import se_resnet50, freeze_backbone_se
from baselines.cbam_resnet import cbam_resnet50, freeze_backbone_cbam
from baselines.eca_resnet import eca_resnet50, freeze_backbone_eca
from our_method.mspa_convnext import MSPAConvNeXt


# ============================================================================
# 配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验全局配置"""
    data_dir: str = './data'
    results_dir: str = './results/tables'
    batch_size: int = 128
    num_workers: int = min(8, multiprocessing.cpu_count())
    seed: int = 42

    # 基线训练协议
    baseline_epochs: int = 30
    baseline_lr: float = 1e-3

    # MSPA 训练协议
    mspa_epochs: int = 60
    mspa_head_lr: float = 1e-3
    mspa_backbone_lr: float = 1e-4

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)


@dataclass
class DatasetSpec:
    """数据集规格"""
    name: str
    num_classes: int
    superclass_count: int  # CIFAR-100 的超类数, Flowers 用 0 表示无
    image_size: int = 224
    train_transform: Callable = None
    test_transform: Callable = None

    def __post_init__(self):
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            norm,
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            norm,
        ])


# 数据集注册表
DATASET_SPECS: Dict[str, DatasetSpec] = {
    'cifar100': DatasetSpec('cifar100', num_classes=100, superclass_count=20),
    'flowers102': DatasetSpec('flowers102', num_classes=102, superclass_count=0),
}

# CIFAR-100 超类映射
CIFAR100_SUPERCLASS: Dict[int, str] = {
    0: 'aquatic mammals', 1: 'fish', 2: 'flowers', 3: 'food containers',
    4: 'fruit and vegetables', 5: 'household electrical devices',
    6: 'household furniture', 7: 'insects', 8: 'large carnivores',
    9: 'large man-made outdoor things', 10: 'large natural outdoor scenes',
    11: 'large omnivores and herbivores', 12: 'medium-sized mammals',
    13: 'non-insect invertebrates', 14: 'people', 15: 'reptiles',
    16: 'small mammals', 17: 'trees', 18: 'vehicles 1', 19: 'vehicles 2',
}
CIFAR100_SUPERCLASS_MAP: Dict[int, int] = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 18,
    10: 17, 11: 3, 12: 3, 13: 14, 14: 9, 15: 8, 16: 11, 17: 17,
    18: 10, 19: 10, 20: 10, 21: 5, 22: 8, 23: 9, 24: 9, 25: 5,
    26: 16, 27: 16, 28: 16, 29: 16, 30: 0, 31: 0, 32: 0, 33: 11,
    34: 11, 35: 11, 36: 7, 37: 7, 38: 15, 39: 15, 40: 6, 41: 6,
    42: 10, 43: 10, 44: 10, 45: 14, 46: 14, 47: 12, 48: 12,
    49: 12, 50: 12, 51: 6, 52: 6, 53: 5, 54: 5, 55: 0, 56: 0,
    57: 1, 58: 18, 59: 18, 60: 17, 61: 17, 62: 17, 63: 5, 64: 5,
    65: 1, 66: 15, 67: 15, 68: 2, 69: 2, 70: 2, 71: 2, 72: 3,
    73: 3, 74: 3, 75: 19, 76: 19, 77: 19, 78: 13, 79: 13,
    80: 13, 81: 13, 82: 13, 83: 16, 84: 16, 85: 16, 86: 18,
    87: 18, 88: 7, 89: 7, 90: 19, 91: 19, 92: 2, 93: 2,
    94: 1, 95: 1, 96: 14, 97: 14, 98: 11, 99: 11,
}


# ============================================================================
# 设备
# ============================================================================

def get_device() -> torch.device:
    """获取可用设备, GPU 可用时启用 cudnn benchmark"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f'[DEVICE] {torch.cuda.get_device_name(0)}')
    print(f'[DEVICE] Using {device}')
    return device


# ============================================================================
# 数据管道
# ============================================================================

class CutMixCollator:
    """CutMix 数据增强 collator (用于 MSPA 训练)"""

    def __init__(self, alpha: float = 1.0, num_classes: int = 100):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(self, batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        B = images.size(0)

        # 随机选择混合比例
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        lam = max(lam, 1.0 - lam)

        # 随机排列
        index = torch.randperm(B)

        # 混合图像
        bx1, by1, bx2, by2 = _rand_bbox(images.shape[-2:], lam)
        images[:, :, bx1:bx2, by1:by2] = images[index, :, bx1:bx2, by1:by2]

        # 调整 lambda 为实际混合比例
        lam = 1 - ((bx2 - bx1) * (by2 - by1) / (images.size(-2) * images.size(-1)))
        return images, labels, labels[index], lam


def _rand_bbox(size: Tuple[int, int], lam: float) -> Tuple[int, int, int, int]:
    """生成 CutMix 的随机裁剪区域"""
    H, W = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(0, H), np.random.randint(0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, x1, y2, x2


class CutMixLoss(nn.Module):
    """CutMix 损失: lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)"""

    def forward(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        return lam * nn.functional.cross_entropy(pred, y_a) + \
               (1.0 - lam) * nn.functional.cross_entropy(pred, y_b)


def create_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    use_cutmix: bool = False,
    num_classes: int = 100,
) -> Dict[str, DataLoader]:
    """
    创建训练/验证/测试 DataLoader.

    Args:
        dataset_name: 'cifar100' 或 'flowers102'
        data_dir: 数据存储根目录
        batch_size: 批次大小
        num_workers: 数据加载进程数
        use_cutmix: 是否使用 CutMix 增强
        num_classes: 类别数 (用于 CutMix collator)

    Returns:
        {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    spec = DATASET_SPECS[dataset_name]

    # 选择数据集类
    if dataset_name == 'cifar100':
        dataset_cls = datasets.CIFAR100
    elif dataset_name == 'flowers102':
        dataset_cls = datasets.Flowers102
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    # 加载完整训练集
    if dataset_name == 'flowers102':
        full_train = dataset_cls(
            root=data_dir, split='train', transform=spec.train_transform, download=True,
        )
    else:
        full_train = dataset_cls(
            root=data_dir, train=True, transform=spec.train_transform, download=True,
        )

    # 数据集切分
    if hasattr(full_train, 'targets'):
        targets = full_train.targets
    elif hasattr(full_train, '_labels'):
        targets = full_train._labels
    else:
        targets = [full_train[i][1] for i in range(len(full_train))]  # fallback, slow

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 测试集
    if dataset_name == 'flowers102':
        test_set = dataset_cls(
            root=data_dir, split='test', transform=spec.test_transform, download=True,
        )
    else:
        test_set = dataset_cls(
            root=data_dir, train=False, transform=spec.test_transform, download=True,
        )

    # DataLoader
    collate_fn = CutMixCollator(num_classes=num_classes) if use_cutmix else None
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, drop_last=use_cutmix,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'[DATA] {dataset_name}: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}')
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


# ============================================================================
# 模型工厂
# ============================================================================

def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """
    统一替换分类头, 处理 torchvision 各模型 API 差异.

    覆盖:
      - ResNet / DenseNet / ShuffleNet: .fc
      - MobileNet / EfficientNet / ConvNeXt: .classifier (Sequential)
      - ViT: .heads (Sequential)
      - Swin: .head
    """
    # Sequential classifier
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        layers = list(model.classifier)
        for i in range(len(layers) - 1, -1, -1):
            if isinstance(layers[i], nn.Linear):
                layers[i] = nn.Linear(layers[i].in_features, num_classes)
                model.classifier = nn.Sequential(*layers)
                return model
        raise ValueError('No Linear layer found in classifier Sequential')

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    # Simple Linear head
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # ViT .heads (Sequential)
    if hasattr(model, 'heads') and isinstance(model.heads, nn.Sequential):
        for name, child in model.heads.named_children():
            if isinstance(child, nn.Linear):
                setattr(model.heads, name, nn.Linear(child.in_features, num_classes))
                return model

    if hasattr(model, 'heads') and isinstance(model.heads, nn.Linear):
        model.heads = nn.Linear(model.heads.in_features, num_classes)
        return model

    # Swin .head
    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f'Cannot locate classifier head for {type(model).__name__}')


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    冻结骨干, 仅训练分类头.

    分类头参数名的 key 特征: 'fc.', 'classifier.', 'heads.', 'head.'
    """
    head_keys = ('fc.', 'classifier.', 'heads.', 'head.')
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in head_keys)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f'  [FREEZE] {trainable}/{total} parameter groups trainable')
    return model


# --- 模型注册表 ---

def _build_baseline(model_name: str, num_classes: int) -> nn.Module:
    """构建标准 torchvision 基线模型"""
    builders = {
        'resnet50':        lambda: resnet50(weights=ResNet50_Weights.DEFAULT),
        'densenet121':     lambda: densenet121(weights=DenseNet121_Weights.DEFAULT),
        'mobilenetv3':     lambda: mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT),
        'efficientnet_b2': lambda: efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT),
        'shufflenetv2':    lambda: shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT),
        'convnext_tiny':   lambda: convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT),
        'vit_b16':         lambda: vit_b_16(weights=ViT_B_16_Weights.DEFAULT),
        'swin_t':          lambda: swin_t(weights=Swin_T_Weights.DEFAULT),
    }
    if model_name not in builders:
        raise ValueError(f'Unknown baseline: {model_name}')
    model = builders[model_name]()
    model = replace_classifier(model, num_classes)
    model = freeze_backbone(model)
    return model


def _build_attention_baseline(model_name: str, num_classes: int) -> nn.Module:
    """构建注意力基线模型 (SE/CBAM/ECA on ResNet-50)"""
    builders = {
        'se_resnet50':  (se_resnet50, freeze_backbone_se),
        'cbam_resnet50': (cbam_resnet50, freeze_backbone_cbam),
        'eca_resnet50': (eca_resnet50, freeze_backbone_eca),
    }
    if model_name not in builders:
        raise ValueError(f'Unknown attention baseline: {model_name}')
    build_fn, freeze_fn = builders[model_name]
    model = build_fn(num_classes=num_classes, pretrained=True)
    model = freeze_fn(model)
    return model


def _build_mspa(model_name: str, num_classes: int) -> nn.Module:
    """构建 MSPA-ConvNeXt 创新方法"""
    model = MSPAConvNeXt(
        num_classes=num_classes,
        insert_stages=(2, 3),
        unfreeze_stages=2,
    )
    total, trainable = _count_params(model)
    print(f'  [MSPA] Total: {total/1e6:.2f}M, Trainable: {trainable/1e3:.1f}K')
    return model


def _count_params(model: nn.Module) -> Tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _build_convnext_finetuned(model_name: str, num_classes: int) -> nn.Module:
    """ConvNeXt-Tiny 公平微调基线: 与 MSPA 相同协议但无 MSPA 模块.

    关键: 用于隔离 MSPA 模块的真实贡献。
    协议: 解冻最后 2 个 stage + 分层 LR + 60 epoch + CutMix + WarmRestarts.
    """
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
    model = replace_classifier(model, num_classes)

    # 冻结全部 features
    for p in model.features.parameters():
        p.requires_grad = False

    # 解冻最后 2 个 stage — 与 MSPAConvNeXt(unfreeze_stages=2) 一致
    # features 结构: [0]=stem, [1]=S0, [2]=down, [3]=S1, [4]=down, [5]=S2, [6]=down, [7]=S3
    features = model.features
    start_idx = len(features) - 2 * 2
    start_idx = max(0, start_idx)
    for i in range(start_idx, len(features)):
        for p in features[i].parameters():
            p.requires_grad = True

    # Monkey-patch get_param_groups 以支持分层学习率
    def get_param_groups(base_lr=1e-3, backbone_lr=1e-4):
        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        groups = [{'params': head_params, 'lr': base_lr}]
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': backbone_lr})
        return groups

    model.get_param_groups = get_param_groups

    total, trainable = _count_params(model)
    print(f'  [CONVNEXT-FT] Total: {total/1e6:.2f}M, Trainable: {trainable/1e3:.1f}K')
    return model


MODEL_REGISTRY: Dict[str, Callable] = {
    'resnet50':          _build_baseline,
    'densenet121':       _build_baseline,
    'mobilenetv3':       _build_baseline,
    'efficientnet_b2':   _build_baseline,
    'shufflenetv2':      _build_baseline,
    'convnext_tiny':     _build_baseline,
    'convnext_tiny_ft':  _build_convnext_finetuned,
    'vit_b16':           _build_baseline,
    'swin_t':            _build_baseline,
    'se_resnet50':       _build_attention_baseline,
    'cbam_resnet50':     _build_attention_baseline,
    'eca_resnet50':      _build_attention_baseline,
    'mspa_convnext':     _build_mspa,
}


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """统一模型构建接口"""
    builder = MODEL_REGISTRY.get(model_name)
    if builder is None:
        raise ValueError(
            f'Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}'
        )
    model = builder(model_name, num_classes)
    return model.to(device)


# ============================================================================
# 训练引擎
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int], List[float]]:
    """
    评估模型。

    Returns:
        top1_acc: Top-1 准确率
        top5_acc: Top-5 准确率
        all_preds: 所有预测标签
        all_labels: 所有真实标签
        all_probs: 所有预测概率 (用于置信度分析)
    """
    model.eval()
    correct_top1 = correct_top5 = total = 0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)

        # Top-1
        preds = logits.argmax(dim=1)
        correct_top1 += (preds == labels).sum().item()

        # Top-5
        _, top5_preds = logits.topk(5, dim=1)
        correct_top5 += sum(1 for i in range(len(labels)) if labels[i] in top5_preds[i])

        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(logits.softmax(dim=1).cpu().tolist())

    return (
        correct_top1 / total,
        correct_top5 / total,
        all_preds,
        all_labels,
        all_probs,
    )


def compute_metrics(
    preds: List[int],
    labels: List[int],
    num_classes: int,
    superclass_map: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """
    计算完整评估指标。

    Returns:
        accuracy, top5_accuracy (需外部提供),
        macro_precision, macro_recall, macro_f1,
        per_class_f1 (list),
        superclass_accuracy (如果有 superclass_map)
    """
    # 逐类统计
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for p, l in zip(preds, labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1

    f1s = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    n = len(labels)
    metrics = {
        'accuracy': sum(tp.values()) / n if n > 0 else 0.0,
        'macro_f1': sum(f1s) / len(f1s),
    }

    # 超类准确率 (仅 CIFAR-100)
    if superclass_map is not None:
        superclass_correct = Counter()
        superclass_total = Counter()
        for p, l in zip(preds, labels):
            sc = superclass_map.get(l)
            if sc is not None:
                superclass_total[sc] += 1
                if superclass_map.get(p) == sc:
                    superclass_correct[sc] += 1
        sc_accs = [
            superclass_correct[sc] / superclass_total[sc]
            if superclass_total[sc] > 0 else 0.0
            for sc in range(max(superclass_map.values()) + 1)
        ]
        metrics['superclass_accuracy'] = sum(sc_accs) / len(sc_accs)

    return metrics


def train_epoch_standard(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[object] = None,
) -> float:
    """标准训练 epoch (无 CutMix)"""
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


def train_epoch_cutmix(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CutMixLoss,
    device: torch.device,
) -> float:
    """CutMix 训练 epoch"""
    model.train()
    total_loss, n = 0.0, 0
    for images, y_a, y_b, lam in loader:
        images = images.to(device)
        y_a = y_a.to(device)
        y_b = y_b.to(device)
        lam = torch.tensor(lam, device=device)
        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, y_a, y_b, lam)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


# ============================================================================
# 实验执行器
# ============================================================================

@dataclass
class ExperimentResult:
    """单次实验结果"""
    model_name: str
    dataset: str
    best_val_acc: float
    test_top1_acc: float
    test_top5_acc: float
    macro_f1: float
    params_total: int
    params_trainable: int
    training_time_min: float
    training_history: List[Dict[str, float]] = field(default_factory=list)


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> ExperimentResult:
    """
    运行单个模型的完整实验: 训练 → 验证 → 测试 → 收集指标。
    """
    spec = DATASET_SPECS[dataset_name]
    is_extended = model_name in ('mspa_convnext', 'convnext_tiny_ft')
    num_epochs = config.mspa_epochs if is_extended else config.baseline_epochs

    print(f'\n{"="*70}')
    print(f'  {model_name}  on  {dataset_name}  ({num_epochs} epochs)')
    print(f'{"="*70}')

    # --- 数据 ---
    loaders = create_dataloaders(
        dataset_name, config.data_dir, config.batch_size, config.num_workers,
        use_cutmix=is_extended, num_classes=spec.num_classes,
    )

    # --- 模型 ---
    model = build_model(model_name, spec.num_classes, device)
    params_total, params_trainable = _count_params(model)

    # --- 优化器和调度器 ---
    if is_extended:
        param_groups = model.get_param_groups(config.mspa_head_lr, config.mspa_backbone_lr)
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.baseline_lr, weight_decay=1e-4,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_cutmix = CutMixLoss()

    # --- 训练循环 ---
    best_val_acc = 0.0
    best_state = None
    history = []

    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()

        # 训练
        if is_extended:
            train_loss = train_epoch_cutmix(model, loaders['train'], optimizer, criterion_cutmix, device)
        else:
            train_loss = train_epoch_standard(
                model, loaders['train'], optimizer, criterion_ce, device,
                scheduler=None if is_extended else None,
            )

        # extended 协议使用 CosineAnnealingWarmRestarts，每个 epoch 步进
        if is_extended and scheduler is not None:
            scheduler.step(epoch - 1)

        # 验证
        val_acc, _, _, _, _ = evaluate(model, loaders['val'], device)

        # 标准协议使用 CosineAnnealingLR，每个 epoch 步进
        if not is_extended and scheduler is not None:
            scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'val_acc': round(val_acc, 6),
            'time_s': round(time.time() - t_ep, 1),
        })

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f'  Epoch {epoch:3d}/{num_epochs} | '
              f'loss={train_loss:.4f} | val_acc={val_acc:.4f} | '
              f'{time.time() - t_ep:.1f}s')

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存模型 checkpoint (仅关键模型，节省磁盘空间)
    key_models = {'convnext_tiny', 'convnext_tiny_ft', 'mspa_convnext'}
    if model_name in key_models:
        ckpt_dir = os.path.join(config.results_dir, '..', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'{model_name}_{dataset_name}.pt')
        torch.save(best_state, ckpt_path)
        print(f'  [CHECKPOINT] Saved to {ckpt_path}')

    # --- 最终测试 ---
    test_top1, test_top5, preds, labels, _ = evaluate(model, loaders['test'], device)
    metrics = compute_metrics(
        preds, labels, spec.num_classes,
        superclass_map=CIFAR100_SUPERCLASS_MAP if dataset_name == 'cifar100' else None,
    )

    elapsed = (time.time() - t_start) / 60.0
    print(f'  => Best Val: {best_val_acc:.4f} | '
          f'Test Top-1: {test_top1:.4f} | Top-5: {test_top5:.4f} | '
          f'Macro F1: {metrics["macro_f1"]:.4f} | Time: {elapsed:.1f}min')

    return ExperimentResult(
        model_name=model_name,
        dataset=dataset_name,
        best_val_acc=round(best_val_acc, 6),
        test_top1_acc=round(test_top1, 6),
        test_top5_acc=round(test_top5, 6),
        macro_f1=round(metrics['macro_f1'], 6),
        params_total=params_total,
        params_trainable=params_trainable,
        training_time_min=round(elapsed, 2),
        training_history=history,
    )


def run_ablation(
    dataset_name: str,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, ExperimentResult]:
    """
    MSPA 消融实验: 逐步移除分支, 验证各组件的贡献。

    变体:
      - mspa_full:   CA + SA + MA + FA (完整 4 分支)
      - mspa_no_fa:  CA + SA + MA        (移除频域)
      - mspa_no_ma:  CA + SA + FA        (移除多尺度)
      - mspa_base:   CA + SA             (仅通道+空间, 退化为 CSPA)
    """
    print(f'\n{"="*70}')
    print(f'  ABLATION STUDY: MSPA Component Analysis on {dataset_name}')
    print(f'{"="*70}')

    spec = DATASET_SPECS[dataset_name]
    results = {}

    # 动态切换分支 — 通过修改模型的 forward 逻辑
    # 简单方案: 用 MSPA 子类控制分支开关
    from our_method.mspa import MSPA, ChannelAttention, SpatialAttention
    from our_method.mspa_convnext import MSPAConvNeXt as _MSPAConvNeXt

    variants = {
        'mspa_full':  {'use_fa': True,  'use_ma': True},
        'mspa_no_fa': {'use_fa': False, 'use_ma': True},
        'mspa_no_ma': {'use_fa': True,  'use_ma': False},
        'mspa_base':  {'use_fa': False, 'use_ma': False},
    }

    for variant_name, flags in variants.items():
        print(f'\n  --- {variant_name}: FA={flags["use_fa"]}, MA={flags["use_ma"]} ---')

        model = _MSPAConvNeXt(
            num_classes=spec.num_classes,
            insert_stages=(2, 3),
            unfreeze_stages=2,
        )

        # 根据消融配置禁用分支
        for mspa_mod in model.mspa_modules.values():
            if not flags['use_fa']:
                mspa_mod.fa = None  # 禁用频域分支
            if not flags['use_ma']:
                mspa_mod.ma = None  # 禁用多尺度分支

        model = model.to(device)
        loaders = create_dataloaders(
            dataset_name, config.data_dir, config.batch_size, config.num_workers,
            use_cutmix=True, num_classes=spec.num_classes,
        )

        param_groups = model.get_param_groups(config.mspa_head_lr, config.mspa_backbone_lr)
        optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
        criterion = CutMixLoss()

        best_val_acc = 0.0
        best_state = None
        num_epochs = config.mspa_epochs

        t_start = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch_cutmix(model, loaders['train'], optimizer, criterion, device)
            scheduler.step(epoch - 1)
            val_acc, _, _, _, _ = evaluate(model, loaders['val'], device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0:
                print(f'    Epoch {epoch}/{num_epochs}: loss={train_loss:.4f}, val_acc={val_acc:.4f}')

        if best_state is not None:
            model.load_state_dict(best_state)

        test_top1, test_top5, preds, labels, _ = evaluate(model, loaders['test'], device)
        metrics = compute_metrics(
            preds, labels, spec.num_classes,
            superclass_map=CIFAR100_SUPERCLASS_MAP if dataset_name == 'cifar100' else None,
        )
        elapsed = (time.time() - t_start) / 60.0

        print(f'  {variant_name}: Best Val={best_val_acc:.4f}, '
              f'Test Top1={test_top1:.4f}, F1={metrics["macro_f1"]:.4f},'
              f'Time={elapsed:.1f}min')

        results[variant_name] = ExperimentResult(
            model_name=variant_name,
            dataset=dataset_name,
            best_val_acc=round(best_val_acc, 6),
            test_top1_acc=round(test_top1, 6),
            test_top5_acc=round(test_top5, 6),
            macro_f1=round(metrics['macro_f1'], 6),
            params_total=_count_params(model)[0],
            params_trainable=_count_params(model)[1],
            training_time_min=round(elapsed, 2),
        )

    return results


# ============================================================================
# 结果输出
# ============================================================================

def save_results(
    results: Dict[str, ExperimentResult],
    config: ExperimentConfig,
    prefix: str = '',
) -> str:
    """保存实验结果到 JSON 文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results_{prefix}_{timestamp}.json' if prefix else f'results_{timestamp}.json'
    save_path = os.path.join(config.results_dir, filename)

    output = {}
    for key, r in results.items():
        output[key] = {
            'model': r.model_name,
            'dataset': r.dataset,
            'best_val_acc': r.best_val_acc,
            'test_top1_acc': r.test_top1_acc,
            'test_top5_acc': r.test_top5_acc,
            'macro_f1': r.macro_f1,
            'params_total': r.params_total,
            'params_trainable': r.params_trainable,
            'training_time_min': r.training_time_min,
            'training_history': r.training_history,
        }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n[SAVED] {save_path}')
    return save_path


def print_summary(results: Dict[str, ExperimentResult]) -> None:
    """打印结果汇总表格"""
    print(f'\n{"="*90}')
    print('RESULTS SUMMARY')
    print(f'{"="*90}')
    header = f'{"Model":<22} {"Dataset":<12} {"Val Acc":>8} {"Test Top1":>10} {"Top5":>8} {"F1":>8} {"Time(min)":>10}'
    print(header)
    print('-' * len(header))
    for r in results.values():
        print(
            f'{r.model_name:<22} {r.dataset:<12} {r.best_val_acc:>8.4f} '
            f'{r.test_top1_acc:>10.4f} {r.test_top5_acc:>8.4f} '
            f'{r.macro_f1:>8.4f} {r.training_time_min:>10.1f}'
        )


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。支持选择性运行模型、数据集、消融实验和冒烟测试。"""
    parser = argparse.ArgumentParser(
        description='图像分类模型对比实验 — 统一实验框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py                                    # 全部实验
  python run_experiments.py --models resnet50 convnext_tiny    # 指定模型
  python run_experiments.py --datasets cifar100                # 指定数据集
  python run_experiments.py --epochs 5 --dry_run               # 冒烟测试
  python run_experiments.py --ablation                         # 仅消融实验
        """,
    )
    parser.add_argument(
        '--models', nargs='+',
        default=list(MODEL_REGISTRY.keys()),
        help='要运行的模型列表 (默认: 全部)',
    )
    parser.add_argument(
        '--datasets', nargs='+', default=['cifar100'],
        choices=['cifar100', 'flowers102'],
        help='要运行的数据集 (默认: cifar100)',
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认训练轮数')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dry_run', action='store_true',
                        help='冒烟测试模式: 5 epochs, 快速验证')
    parser.add_argument('--ablation', action='store_true',
                        help='仅运行消融实验')
    parser.add_argument('--skip_baselines', action='store_true',
                        help='跳过标准基线, 仅运行注意力基线和创新方法')
    return parser.parse_args()


# ============================================================================
# 主入口
# ============================================================================

def main():
    """主入口：解析参数 → 构建配置 → 运行实验 → 保存结果。"""
    args = parse_args()
    config = ExperimentConfig(batch_size=args.batch_size)

    # 冒烟测试
    if args.dry_run:
        config.baseline_epochs = 5
        config.mspa_epochs = 5
        config.batch_size = 64
        print('[DRY RUN] 5 epochs, batch_size=64')

    # Epoch 覆盖
    if args.epochs is not None:
        config.baseline_epochs = args.epochs
        config.mspa_epochs = args.epochs

    device = get_device()
    config.num_workers = min(8, multiprocessing.cpu_count()) if device.type == 'cuda' else 0

    print(f'[CONFIG] Models: {args.models}')
    print(f'[CONFIG] Datasets: {args.datasets}')
    print(f'[CONFIG] Baseline epochs: {config.baseline_epochs}, MSPA epochs: {config.mspa_epochs}')
    print(f'[CONFIG] Batch size: {config.batch_size}, Workers: {config.num_workers}')

    all_results: Dict[str, ExperimentResult] = {}

    # --- 消融实验 ---
    if args.ablation:
        for ds in args.datasets:
            ablation_results = run_ablation(ds, config, device)
            all_results.update(ablation_results)
        save_results(all_results, config, prefix='ablation')
        print_summary(all_results)
        return

    # --- 完整实验 ---
    models_to_run = args.models
    if args.skip_baselines:
        standard_baselines = [
            'resnet50', 'densenet121', 'mobilenetv3', 'efficientnet_b2',
            'shufflenetv2', 'convnext_tiny', 'vit_b16', 'swin_t',
        ]
        models_to_run = [m for m in models_to_run if m not in standard_baselines]
        print(f'[SKIP] Standard baselines excluded. Running: {models_to_run}')

    for ds in args.datasets:
        for model_name in models_to_run:
            t_start = time.time()
            try:
                result = run_single_experiment(model_name, ds, config, device)
                all_results[f'{model_name}_{ds}'] = result
                # 每个模型完成后立即保存中间结果，防止崩溃丢失
                save_results(all_results, config, prefix='checkpoint')
            except Exception as e:
                print(f'[ERROR] {model_name} on {ds}: {e}')
                import traceback
                traceback.print_exc()
            print(f'  [{(time.time() - t_start) / 60:.1f} min total for {model_name}]')

    # --- 保存 ---
    save_results(all_results, config)
    print_summary(all_results)
    print('\n[DONE] All experiments completed.')


if __name__ == '__main__':
    main()

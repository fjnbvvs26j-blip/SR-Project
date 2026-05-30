"""
ECA-ResNet50: Efficient Channel Attention (Qilong Wang et al., CVPR 2020)

ECA: GAP → 自适应核大小 1D Conv → Sigmoid, 参数增量几乎为 0。
训练策略: 冻结 ResNet 原始权重, 训练 ECA 模块 + 分类头。
"""
import math
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ECALayer(nn.Module):
    """Efficient Channel Attention: 自适应核大小的 1D 卷积"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)                          # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B, 1, C]
        y = self.conv1d(y)                        # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)    # [B, C, 1, 1]
        return x * self.sigmoid(y)


class ECABottleneck(nn.Module):
    """ResNet Bottleneck + ECA after 3x3 conv"""
    def __init__(self, orig):
        super().__init__()
        self.conv1 = orig.conv1
        self.bn1 = orig.bn1
        self.conv2 = orig.conv2
        self.bn2 = orig.bn2
        self.conv3 = orig.conv3
        self.bn3 = orig.bn3
        self.relu = orig.relu
        self.downsample = orig.downsample
        self.eca = ECALayer(orig.bn3.num_features)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.eca(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _inject_eca(model):
    for name, child in model.named_children():
        if child.__class__.__name__ == 'Bottleneck':
            setattr(model, name, ECABottleneck(child))
        else:
            _inject_eca(child)


def eca_resnet50(num_classes=100, pretrained=True):
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    _inject_eca(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone_eca(model):
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ('eca.', 'fc.'))
    return model


if __name__ == '__main__':
    m = eca_resnet50(100)
    m = freeze_backbone_eca(m)
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'ECA-ResNet50: total={t/1e6:.2f}M, trainable={tr/1e3:.1f}K')
    x = torch.randn(2, 3, 224, 224)
    print(f'Output: {m(x).shape}')

"""
CBAM-ResNet50: Convolutional Block Attention Module (Sanghyun Woo et al., ECCV 2018)

串行结构: 通道注意力 → 空间注意力, 插入 ResNet-50 每个 Bottleneck 后。
训练策略: 冻结 ResNet 原始权重, 训练 CBAM 模块 + 分类头。
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ChannelGate(nn.Module):
    """CBAM 通道注意力: GAP + GMP → 共享 MLP → 相加 → Sigmoid"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(B, C))
        max = self.mlp(self.max_pool(x).view(B, C))
        return self.sigmoid(avg + max).view(B, C, 1, 1)


class SpatialGate(nn.Module):
    """CBAM 空间注意力: 通道压缩 → 7x7 Conv → Sigmoid"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max = x.max(dim=1, keepdim=True)[0]
        return self.sigmoid(self.conv(torch.cat([avg, max], dim=1)))


class CBAM(nn.Module):
    """CBAM: 串行 — 先通道注意力, 再空间注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(channels, reduction)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x


class CBAMBottleneck(nn.Module):
    """ResNet Bottleneck + CBAM after 3x3 conv"""
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
        self.cbam = CBAM(orig.bn3.num_features)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _inject_cbam(model):
    for name, child in model.named_children():
        if child.__class__.__name__ == 'Bottleneck':
            setattr(model, name, CBAMBottleneck(child))
        else:
            _inject_cbam(child)


def cbam_resnet50(num_classes=100, pretrained=True):
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    _inject_cbam(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone_cbam(model):
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ('cbam.', 'fc.'))
    return model


if __name__ == '__main__':
    m = cbam_resnet50(100)
    m = freeze_backbone_cbam(m)
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'CBAM-ResNet50: total={t/1e6:.2f}M, trainable={tr/1e3:.1f}K')
    x = torch.randn(2, 3, 224, 224)
    print(f'Output: {m(x).shape}')

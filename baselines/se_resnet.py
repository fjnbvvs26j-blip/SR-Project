"""
SE-ResNet50: Squeeze-and-Excitation 增强 ResNet-50 (Jie Hu et al., CVPR 2018)

在 ResNet-50 每个 Bottleneck 的 3x3 conv 后插入 SE 模块。
训练策略: 冻结 ResNet 原始权重, 训练 SE 模块 + 分类头。
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SELayer(nn.Module):
    """Squeeze-and-Excitation: GAP → FC → ReLU → FC → Sigmoid"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class SEBottleneck(nn.Module):
    """ResNet Bottleneck + SE module after 3x3 conv"""
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
        self.se = SELayer(orig.bn3.num_features)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def _inject_se(model):
    """递归遍历 ResNet, 将每个 Bottleneck 替换为 SEBottleneck"""
    for name, child in model.named_children():
        if child.__class__.__name__ == 'Bottleneck':
            setattr(model, name, SEBottleneck(child))
        else:
            _inject_se(child)


def se_resnet50(num_classes=100, pretrained=True):
    """构建 SE-ResNet50, 返回 (model, 冻结函数)"""
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    _inject_se(model)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone_se(model):
    """冻结除 SE 模块和分类头外的所有参数"""
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ('se.', 'fc.'))
    return model


# test
if __name__ == '__main__':
    m = se_resnet50(100)
    m = freeze_backbone_se(m)
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'SE-ResNet50: total={t/1e6:.2f}M, trainable={tr/1e3:.1f}K')
    x = torch.randn(2, 3, 224, 224)
    print(f'Output: {m(x).shape}')

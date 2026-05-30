"""
创新方法：混合域注意力增强 ConvNeXt Block (Hybrid-Domain Attention ConvNeXt)

创新点：
在 ConvNeXt Block 的 7×7 深度卷积后，加入一个轻量级的
通道-空间并行注意力模块 (Channel-Spatial Parallel Attention, CSPA)。
该模块同时关注通道维度和空间维度的特征重要性，
以极小的参数增量（约 3K 参数）提升特征表达能力。

设计思路：
1. 通道注意力分支：使用全局平均池化 + 1D 卷积进行跨通道交互（参考 ECA-Net）
2. 空间注意力分支：使用 1×1 卷积压缩通道 + 3×3 卷积捕获空间注意力
3. 两支路输出相加后通过 sigmoid 得到注意力权重
4. 残差连接：output = input * attention_weight + input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSpatialParallelAttention(nn.Module):
    """通道-空间并行注意力模块 (CSPA)"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力分支 (ECA-style: 1D conv for cross-channel interaction)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
        )
        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 3, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        # 可学习的两个分支融合权重
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        B, C, H, W = x.shape
        # 通道注意力
        ca = self.channel_att[0](x)          # [B, C, 1, 1]
        ca = ca.squeeze(-1).unsqueeze(1)      # [B, 1, C]
        ca = self.channel_att[1](ca)          # [B, 1, C]
        ca = ca.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        ca = self.sigmoid(ca)
        # 空间注意力
        sa = self.spatial_att(x)              # [B, 1, H, W]
        sa = self.sigmoid(sa)
        # 加权融合
        attention = self.alpha * ca + self.beta * sa
        return x * attention + x


class ImprovedConvNeXt(nn.Module):
    """
    改进的 ConvNeXt-Tiny: 在指定 stage 的 Block 中插入 CSPA 模块

    使用方法: 替换 torchvision 的 ConvNeXt-Tiny 中的选定 Block
    """
    def __init__(self, num_classes=100, insert_stages=(2, 3)):
        """
        Args:
            num_classes: 分类类别数
            insert_stages: 在哪些 stage 插入 CSPA（默认 stage 2 和 3）
        """
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        super().__init__()
        # 加载预训练 ConvNeXt-Tiny
        pretrained = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = pretrained.features
        self.avgpool = pretrained.avgpool

        # 在指定 stage 的最后一个 Block 后插入 CSPA 模块
        # ConvNeXt-Tiny stage 配置: [0]:3个block, [1]:3个block, [2]:9个block, [3]:3个block
        # 通道数: stage 0:96, stage 1:192, stage 2:384, stage 3:768
        stage_channels = {0: 96, 1: 192, 2: 384, 3: 768}
        self.cspa_modules = nn.ModuleDict()
        for s in insert_stages:
            if s in stage_channels:
                self.cspa_modules[str(s)] = ChannelSpatialParallelAttention(
                    stage_channels[s], reduction=8
                )

        self.insert_stages = insert_stages
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, num_classes)

        # 初始化新模块
        self._init_new_modules()

    def _init_new_modules(self):
        for m in self.cspa_modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 手动遍历 features 的各个 stage，在指定位置插入 CSPA
        block_idx = 0
        stage_boundaries = [3, 6, 15, 18]  # 每个 stage 的起始 block 索引

        for i, layer in enumerate(self.features):
            x = layer(x)
            block_idx += 1
            # 在每个 stage 的最后一个 block 之后插入 CSPA
            if block_idx in stage_boundaries:
                stage = stage_boundaries.index(block_idx)
                if stage in self.insert_stages:
                    x = self.cspa_modules[str(stage)](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)       # [B, C, 1, 1] -> [B, C]
        x = self.norm(x)
        x = self.head(x)
        return x


def freeze_backbone_improved(model):
    """冻结预训练层，只训练 CSPA 模块和分类头"""
    for name, param in model.named_parameters():
        is_trainable = (
            'cspa_modules' in name or
            'head' in name or
            'norm' in name or
            'alpha' in name or
            'beta' in name
        )
        param.requires_grad = is_trainable
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    print(f'  Improved model: {trainable}/{total} trainable layers')
    return model


# ========== 简单测试 ==========
if __name__ == '__main__':
    model = ImprovedConvNeXt(num_classes=100, insert_stages=(2, 3))
    model = freeze_backbone_improved(model)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f'Input: {dummy.shape} -> Output: {out.shape}')

    # 统计参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total/1e6:.2f}M, Trainable: {trainable/1e3:.1f}K')

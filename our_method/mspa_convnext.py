"""
MSPA-ConvNeXt: 将 MSPA 模块嵌入 ConvNeXt-Tiny 的指定 stage 后

torchvision 0.15.2 ConvNeXt-Tiny features 结构:
  [0] Conv2dNormActivation  (stem, out=96)
  [1] Sequential x3 CNBlock (stage 0, ch=96, 56x56)
  [2] Sequential LN+Conv2d  (downsample, out=192)
  [3] Sequential x3 CNBlock (stage 1, ch=192, 28x28)
  [4] Sequential LN+Conv2d  (downsample, out=384)
  [5] Sequential x9 CNBlock (stage 2, ch=384, 14x14)  ← MSPA after this
  [6] Sequential LN+Conv2d  (downsample, out=768)
  [7] Sequential x3 CNBlock (stage 3, ch=768, 7x7)    ← MSPA after this

训练策略:
  - 冻结 ConvNeXt 骨干 (可选: 解冻最后 n 个 stage)
  - 训练所有 MSPA 模块 + 分类头
  - 分层学习率: 骨干 1e-4, MSPA+分类头 1e-3
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from our_method.mspa import MSPA


class MSPAConvNeXt(nn.Module):
    """ConvNeXt-Tiny + MSPA 多尺度并行注意力

    Args:
        num_classes: 分类类别数
        insert_stages: 在哪些 stage 后插入 MSPA (默认 stage 2, 3)
        unfreeze_stages: 解冻最后几个 stage 的参数 (0=全部冻结)
    """

    # torchvision 0.15.2 features 索引 → stage 映射
    # stage i 的 CNBlocks 在 features[stage_block_idx[i]]
    STAGE_BLOCK_IDX = {0: 1, 1: 3, 2: 5, 3: 7}
    STAGE_CHANNELS = {0: 96, 1: 192, 2: 384, 3: 768}

    def __init__(self, num_classes=100, insert_stages=(2, 3), unfreeze_stages=2):
        super().__init__()

        pretrained = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = pretrained.features
        self.avgpool = pretrained.avgpool

        # 插入 MSPA 的位置: features 索引 → stage 名称
        self.mspa_positions = {}
        self.mspa_modules = nn.ModuleDict()
        for s in insert_stages:
            idx = self.STAGE_BLOCK_IDX[s]
            self.mspa_positions[idx] = str(s)
            self.mspa_modules[str(s)] = MSPA(self.STAGE_CHANNELS[s])

        # 分类头
        self.norm = nn.LayerNorm(768, eps=1e-6)
        self.head = nn.Linear(768, num_classes)

        self._init_weights()
        self._setup_freezing(unfreeze_stages)

    def _init_weights(self):
        for m in self.mspa_modules.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def _setup_freezing(self, unfreeze_stages):
        """冻结骨干，只训练 MSPA + head + norm (+ 可选解冻的 stage)"""
        # 全部冻结
        for p in self.features.parameters():
            p.requires_grad = False

        # 解冻最后 unfreeze_stages 个 stage
        # Stage 2 = features[4]+features[5], Stage 3 = features[6]+features[7]
        # 解冻 1 个 stage → features[6:] (2 elements)
        # 解冻 2 个 stage → features[4:] (4 elements)
        if unfreeze_stages > 0:
            start_idx = len(self.features) - 2 * unfreeze_stages
            start_idx = max(0, start_idx)
            for i in range(start_idx, len(self.features)):
                for p in self.features[i].parameters():
                    p.requires_grad = True

        # MSPA, head, norm 始终可训练
        for p in self.mspa_modules.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        for p in self.norm.parameters():
            p.requires_grad = True

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.mspa_positions:
                s = self.mspa_positions[i]
                x = self.mspa_modules[s](x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.norm(x)
        x = self.head(x)
        return x

    def get_param_groups(self, base_lr=1e-3, backbone_lr=1e-4):
        """分层学习率: MSPA+head+norm=base_lr, 骨干=backbone_lr"""
        mspa_head_params = []
        backbone_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ('mspa_modules', 'head', 'norm',
                                        'alpha', 'beta', 'gamma', 'delta')):
                mspa_head_params.append(param)
            else:
                backbone_params.append(param)

        groups = [{'params': mspa_head_params, 'lr': base_lr}]
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': backbone_lr})
        return groups


def count_trainable_params(model):
    """返回 (总参数量, 可训练参数量) 统计。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ========== 快速测试 ==========
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSPAConvNeXt(num_classes=100, insert_stages=(2, 3), unfreeze_stages=2)
    model = model.to(device)

    total, trainable = count_trainable_params(model)
    print(f'Total params: {total/1e6:.2f}M')
    print(f'Trainable params: {trainable/1e3:.1f}K ({trainable/total*100:.2f}%)')

    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print(f'Input: {dummy.shape} -> Output: {out.shape}')

    from our_method.mspa import count_mspa_params
    for s in (2, 3):
        p = count_mspa_params({2: 384, 3: 768}[s])
        print(f'MSPA stage {s} params: {p/1e3:.2f}K')

    print('All tests passed!')

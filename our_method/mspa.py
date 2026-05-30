"""
MSPA (Multi-Scale Parallel Attention) — 四分支并行注意力模块

分支:
  CA  — 通道注意力 (ECA-Net 风格, 自适应核大小)
  SA  — 空间注意力 (1x1 压缩 + 3x3 卷积)
  MA  — 多尺度空间注意力 (3x3, 5x5, 7x7 三尺度并行)
  FA  — 频域注意力 (FFT + 逐通道可学习复数权重 + IFFT)

融合: 可学习参数 α,β,γ,δ 经 Softmax 归一化, 残差连接

参数增量: ~52K (C=384) + ~205K (C=768) ≈ 257K (两个 MSPA 模块合计), 相对 29M 骨干可忽略
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """ECA-style 通道注意力 (自适应 1D 卷积核)"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        t = int(abs(math.log2(channels) / gamma + b / gamma))
        kernel_size = t if t % 2 == 1 else t + 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.gap(x)                        # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)    # [B, 1, C]
        y = self.conv1d(y)                      # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return self.sigmoid(y)


class SpatialAttention(nn.Module):
    """空间注意力: 1x1 压缩通道 + 3x3 捕获空间依赖"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 3, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))  # [B, 1, H, W]


class MultiScaleAttention(nn.Module):
    """多尺度空间注意力: {3,5,7} 三尺度并行, 取均值"""
    def __init__(self, channels, scales=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, 1, k, padding=k // 2, bias=False)
            for k in scales
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = None
        for conv in self.convs:
            a = self.sigmoid(conv(x))  # [B, 1, H, W]
            attn = a if attn is None else attn + a
        return attn / len(self.convs)  # [B, 1, H, W]


class FrequencyAttention(nn.Module):
    """频域注意力: FFT → 逐通道复数权重 → IFFT → 幅值 → Sigmoid"""
    def __init__(self, channels):
        super().__init__()
        # 逐通道可学习复数权重 (real + imag)
        self.w_real = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.w_imag = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W] real
        Xf = torch.fft.rfft2(x, norm='ortho')          # [B, C, H, W//2+1] complex
        w = torch.complex(self.w_real, self.w_imag)     # [1, C, 1, 1]
        Xf = Xf * w                                     # 逐通道频域加权
        y = torch.fft.irfft2(Xf, s=x.shape[-2:], norm='ortho')  # [B, C, H, W] real
        return self.sigmoid(y)                          # [B, C, H, W]


class MSPA(nn.Module):
    """
    Multi-Scale Parallel Attention — 四分支并行注意力

    A_MSPA(X) = softmax(α,β,γ,δ) · [CA, SA, MA, FA]
    Y = A_MSPA(X) ⊙ X + X
    """
    def __init__(self, channels, reduction=8, scales=(3, 5, 7)):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(channels, reduction)
        self.ma = MultiScaleAttention(channels, scales)
        self.fa = FrequencyAttention(channels)

        # 可学习融合权重, 初始化为相等 (经 softmax 后各 0.25)
        raw = math.log(0.25)
        self.alpha = nn.Parameter(torch.tensor(raw))
        self.beta  = nn.Parameter(torch.tensor(raw))
        self.gamma = nn.Parameter(torch.tensor(raw))
        self.delta = nn.Parameter(torch.tensor(raw))

    def forward(self, x):
        # 计算各分支注意力图 (None 表示该分支被消融实验禁用)
        ca = self.ca(x)                              # [B, C, 1, 1]
        sa = self.sa(x)                              # [B, 1, H, W]
        ma = self.ma(x) if self.ma is not None else None  # [B, 1, H, W] or None
        fa = self.fa(x) if self.fa is not None else None  # [B, C, H, W] or None

        # 收集活跃分支的权重和注意力图
        branches = [
            (self.alpha, ca),
            (self.beta,  sa),
            (self.gamma, ma),
            (self.delta, fa),
        ]
        active_logits = torch.stack([w for w, a in branches if a is not None])
        active_weights = F.softmax(active_logits, dim=0)

        # 加权融合
        attention = None
        idx = 0
        for _, attn_map in branches:
            if attn_map is not None:
                contribution = active_weights[idx] * attn_map
                attention = contribution if attention is None else attention + contribution
                idx += 1

        return x * attention + x


def count_mspa_params(channels, reduction=8, scales=(3, 5, 7)):
    """统计 MSPA 模块参数量 (不含骨干网络)"""
    m = MSPA(channels, reduction, scales)
    return sum(p.numel() for p in m.parameters())

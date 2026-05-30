#!/bin/bash
# Smoke test: verify all models and data pipeline on GPU
set -e
PY=/root/miniconda3/bin/python
cd /root

echo "=== Smoke Test ==="
echo "PyTorch: $($PY -c 'import torch; print(torch.__version__)')"
echo "GPU: $($PY -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

echo "=== Test Data Loading ==="
$PY -c "
import sys; sys.path.insert(0, '.')
from run_experiments import create_dataloaders, get_device
device = get_device()
loaders = create_dataloaders('cifar100', './data', 32, 2)
x, y = next(iter(loaders['train']))
print(f'Sample batch: {x.shape}, labels: {y.shape}')
print('[OK] Data pipeline works')
" 2>&1

echo ""
echo "=== Test All Models (1 forward pass each) ==="
$PY -c "
import sys; sys.path.insert(0, '.')
from run_experiments import build_model, get_device
import torch

device = get_device()
models = [
    'resnet50', 'densenet121', 'mobilenetv3', 'efficientnet_b2',
    'shufflenetv2', 'convnext_tiny', 'vit_b16', 'swin_t',
    'se_resnet50', 'cbam_resnet50', 'eca_resnet50', 'mspa_convnext',
]

dummy = torch.randn(4, 3, 224, 224).to(device)
for name in models:
    print(f'  {name}...', end=' ', flush=True)
    model = build_model(name, 100, device)
    with torch.no_grad():
        out = model(dummy)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'out={list(out.shape)}, total={params/1e6:.1f}M, trainable={trainable/1e3:.1f}K')
    del model
    torch.cuda.empty_cache()

print('[OK] All models pass!')
" 2>&1

echo ""
echo "=== Smoke Test Complete ==="

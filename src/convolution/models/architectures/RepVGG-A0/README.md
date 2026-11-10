# RepVGG-A0

This directory contains a PyTorch implementation of RepVGG-A0 for image classification and feature extraction.

RepVGG uses structural re-parameterization: during training, it benefits from multi-branch residual and identity topology (`3x3 Conv + BN`, `1x1 Conv + BN`, and `Identity + BN`), while at inference time the branches can be merged into a single plain $3 \times 3$ convolution layer via `switch_to_deploy()`.

## Architecture Details

- **Input**: 3 channels ($224 \times 224$)
- **Stage 0**: 1 layer, stride 2, 48 channels
- **Stage 1**: 2 layers, stride 2, 48 channels
- **Stage 2**: 4 layers, stride 2, 96 channels
- **Stage 3**: 14 layers, stride 2, 192 channels
- **Stage 4**: 1 layer, stride 2, 1280 channels
- **Embedding**: 1280-d (or projected)
- **Classifier**: Linear to `num_classes`

## Usage

```python
from convolution.models.architectures.RepVGG_A0 import RepVGGA0

model = RepVGGA0(embedding_dim=1280, num_classes=10)

# Multi-branch training mode
logits = model(images)

# Deployment inference conversion
model.eval()
model.switch_to_deploy()
fast_logits = model(images)
```


# ResNet-18

## Overview

This directory contains a PyTorch implementation of ResNet-18 for image classification and feature extraction.

The model follows the standard ResNet layout:

1. `7 x 7` convolutional stem with stride 2.
2. Maximum pooling.
3. Four residual stages containing two `BasicBlock` modules each.
4. Adaptive average pooling.
5. Optional embedding projection and a linear classifier.

Each residual block applies two `3 x 3` convolutions with batch normalization and ReLU activation. When a stage changes spatial size or channel count, its shortcut uses a `1 x 1` convolution with batch normalization.

## Usage

```python
from convolution.models.architectures.ResNet-18.model import ResNet18

model = ResNet18(embedding_dim=512, num_classes=10)

logits = model(images)       # Shape: [batch_size, 10]
embeddings = model.embed(images)  # Shape: [batch_size, 512]
```

The `embedding_dim` argument controls the size returned by `embed()`. A size of 512 uses the final residual feature map directly; other values add a linear projection.

## Files

- `model.py`: complete ResNet-18 model and feature-extraction interface.
- `blocks.py`: residual `BasicBlock` implementation.
- `config.py`: model configuration data.

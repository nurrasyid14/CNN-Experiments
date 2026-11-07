# AlexNet

## Overview

This directory contains a PyTorch implementation of AlexNet for image classification and feature extraction.

AlexNet uses five convolutional layers followed by adaptive pooling and fully connected layers:

1. A large `11 x 11` convolution extracts early spatial features.
2. Maximum pooling and a second convolutional block refine the representation.
3. Three convolutional layers capture higher-level features.
4. Features are pooled to a fixed `6 x 6` spatial size.
5. The feature vector is projected to `embedding_dim`, followed by the classifier.

ReLU activation and dropout are used after the convolutional pooling stages and in the embedding head.

## Usage

```python
from convolution.models.architectures.AlexNet.model import AlexNet

model = AlexNet(embedding_dim=4096, num_classes=10)

logits = model(images)         # Shape: [batch_size, 10]
embeddings = model.embed(images)  # Shape: [batch_size, 4096]
```

The `embedding_dim` argument controls the size returned by `embed()`.

## Files

- `model.py`: AlexNet classifier and feature-extraction interface.
- `blocks.py`: supporting neural-network layer implementations.
- `config.py`: model configuration data.

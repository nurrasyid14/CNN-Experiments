"""Structural re-parameterization building blocks for RepVGG."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def conv_bn_fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse a Conv2d and BatchNorm2d layer into equivalent (weight, bias)."""
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)
    weight = conv.weight * t
    bias = beta - (running_mean * gamma) / std
    return weight, bias


class RepVGGBlock(nn.Module):
    """RepVGG building block with multi-branch training and single-conv inference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.deploy = deploy

        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        identity_out = 0.0
        if self.rbr_identity is not None:
            identity_out = self.rbr_identity(inputs)

        dense_out = self.rbr_dense(inputs)
        one_by_one_out = self.rbr_1x1(inputs)

        return self.nonlinearity(dense_out + one_by_one_out + identity_out)

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the combined 3x3 kernel and bias for deployment."""
        kernel_3x3, bias_3x3 = conv_bn_fuse(self.rbr_dense[0], self.rbr_dense[1])
        kernel_1x1, bias_1x1 = conv_bn_fuse(self.rbr_1x1[0], self.rbr_1x1[1])
        # Pad 1x1 kernel to 3x3
        kernel_1x1_padded = F.pad(kernel_1x1, [1, 1, 1, 1])

        if self.rbr_identity is not None:
            # Identity branch: construct 3x3 identity kernel and fuse with BN
            identity_kernel = torch.zeros(
                self.in_channels,
                self.in_channels // self.groups,
                3,
                3,
                device=kernel_3x3.device,
                dtype=kernel_3x3.dtype,
            )
            for i in range(self.in_channels):
                identity_kernel[i, i % (self.in_channels // self.groups), 1, 1] = 1.0

            running_mean = self.rbr_identity.running_mean
            running_var = self.rbr_identity.running_var
            gamma = self.rbr_identity.weight
            beta = self.rbr_identity.bias
            eps = self.rbr_identity.eps

            std = torch.sqrt(running_var + eps)
            t = (gamma / std).reshape(-1, 1, 1, 1)
            kernel_id = identity_kernel * t
            bias_id = beta - (running_mean * gamma) / std
        else:
            kernel_id = torch.zeros_like(kernel_3x3)
            bias_id = torch.zeros_like(bias_3x3)

        kernel_fused = kernel_3x3 + kernel_1x1_padded + kernel_id
        bias_fused = bias_3x3 + bias_1x1 + bias_id
        return kernel_fused, bias_fused

    def switch_to_deploy(self) -> None:
        """Convert multi-branch block to single 3x3 conv for fast inference."""
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        self.deploy = True


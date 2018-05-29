import torch
import torch.nn as nn
import numpy as np

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False)
        self.bias = nn.Parameter(
                data=torch.zeros(out_channels, dtype=torch.float32),
                requires_grad=True)
        self.maxpool2d = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)
        self.sumpool2d = nn.LPPool1d(1, kernel_size, stride=stride)

    def forward(self, x, m):
        """
        Forward pass for partial convolution
        Args:
            x: (torch.Tensor) input tensor
            m: (torch.Tensor) mask tensor (same dimensions as x)
        Return:
            (torch.Tensor) result of partial convolution
            (torch.Tensor) new mask tensor (same dimensions as result)
        """
        msum = self.sumpool2d(m)
        new_mask = self.maxpool2d(m)
        new_features = self.conv2d(torch.mul(x, m)) / (msum + 1e-12) + self.bias
        return (new_features, new_mask)

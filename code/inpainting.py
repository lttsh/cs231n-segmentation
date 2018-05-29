import torch
import torch.nn as nn
import numpy as np

class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias)
        self.maxpool2d= nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

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
        return (self.conv2d(torch.mul(x, m)), self.maxpool2d(m))

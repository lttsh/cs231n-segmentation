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
        self.sumpool2d = nn.LPPool2d(1, kernel_size, stride=stride)

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


class _PartialConvEncoderBlock(nn.Module):
    """
    Partiall CNN block for the encoder.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        """
        Args:
            in_channels: (int) number of input channels to this block
            out_channels: (int) number of out_channels from this block
            num_conv_layers: (int) number of total Conv2d filters. Must be >= 2
        """
        super().__init__()
        self.partialConv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        self.nonlinearity = nn.LeakyReLU(alpha=0.2)
    def forward(self, x, m):
        """
        Forward pass for the decoder block
        Args:
            x: (torch.Tensor) input tensor to be encoded by block
            m; (torch.Tensor) input mask for the tensor
        Return:
            (torch.Tensor) result of decoding input tensor
            (torch.Tensor) re
        """
        x_, m_ = self.partialConv(x, m)
        x_ = self.nonlinearity(x_)
        return (x_, m_)

class InpaintingEncoder(nn.Module):
    def _init__(self):
        super().__init__()
        # self.enc1 = _PartialConvEncoderBlock(3, )

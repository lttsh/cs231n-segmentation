import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from dataset import CocoStuffDataSet


def el_1(x):
    return x.abs().sum()

def valid_loss(valid_weight, I_out, I_gt, mask):
    """
    Compute the valid loss for inpainting.
    
    Inputs:
    - valid_weight: Scalar giving the weighting for the valid loss.
    - I_out: features of the output image; Tensor of shape (N, C_l, H_l, W_l).
    - I_gt: features of the ground truh image, Tensor with shape (N, C_l, H_l, W_l).
    - mask: binary mask representing the original pixels given to model.
        Tensor with shape (N, C_l, H_l, W_l)
    
    Returns:
    - scalar valid loss
    """
    return valid_weight * el_1(mask * (I_out-I_gt))

def hole_loss(hole_weight, I_out, I_gt, mask):
    """
    Compute the hole loss for inpainting.
    
    Inputs:
    - hole_weight: Scalar giving the weighting for the hole loss.
    - I_out: features of the output image; Tensor of shape (N, C_l, H_l, W_l).
    - I_gt: features of the ground truh image, Tensor with shape (N, C_l, H_l, W_l).
    - mask: binary mask representing the original pixels given to model.
        Tensor with shape (N, C_l, H_l, W_l)
    
    Returns:
    - scalar hole loss
    """
    return hole_weight * el_1((1.0 - mask) * (I_out-I_gt))

def perceptual_loss(perceptual_weight, cnn, I_out, I_gt, I_comp):
    """
    Compute the perceptual_loss loss for inpainting.
    
    Inputs:
    - perceptual_weight: Scalar giving the weighting for the perceptual loss.
    - cnn: model that accepts images as inputs and provides feature embeddings
    - I_out: features of the output image; Tensor of shape (N, C_l, H_l, W_l).
    - I_gt: features of the ground truh image, Tensor with shape (N, C_l, H_l, W_l).
    - I_comp: same as I_out, but with the non-hole pixels directly
        set to ground truth. Tensor with shape (N, C_l, H_l, W_l).
    
    Returns:
    - scalar perceptual_loss loss
    """
    psi_out = cnn(I_out)
    psi_gt = cnn(I_gt)
    psi_comp = cnn(I_comp)
    return perceptual_weight * (el_1(psi_out - psi_gt) + el_1(psi_comp - psi_gt))

def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N, C, H, W = features.size()
    F = features.view(N, C, -1)

    G = torch.matmul(F, F.transpose(1, 2))
    if normalize:
        G /= (C*H*W)
    return G

def style_loss(style_weight, cnn, I_out, I_gt, I_comp):
    """
    Compute the perceptual_loss loss for inpainting.
    
    Inputs:
    - style_weight: Scalar giving the weighting for the style loss.
    - cnn: model that accepts images as inputs and provides feature embeddings
    - I_out: features of the output image; Tensor of shape (N, C_l, H_l, W_l).
    - I_gt: features of the ground truh image, Tensor with shape (N, C_l, H_l, W_l).
    - I_comp: same as I_out, but with the non-hole pixels directly
        set to ground truth. Tensor with shape (N, C_l, H_l, W_l).
    
    Returns:
    - scalar perceptual_loss loss
    """

    G_out = gram_matrix(cnn(I_out), normalize=True)
    G_comp = gram_matrix(cnn(I_comp), normalize=True)
    G_gt = gram_matrix(cnn(I_gt), normalize=True)
    return style_weight * (el_1(G_out - G_gt) + el_1(G_comp - G_gt))

def tv_loss(tv_weight, I_comp):
    """
    Compute total variation loss.
    
    Inputs:
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    - I_comp: same as I_out, but with the non-hole pixels directly
        set to ground truth. Tensor with shape (N, C_l, H_l, W_l).
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for I_comp weighted by tv_weight.
    """
    img = I_comp
    N, C, H, W = img.size()
    down = torch.cat((img[:,:,1:,:], img[:,:,-1,:].view(N, C, 1, W)), dim=2)
    right = torch.cat((img[:,:,:,1:], img[:,:,:,-1].view(N, C, H, 1)), dim=3)
    return tv_weight * (el_1(down - img) + el_1(right - img))

def total_loss(loss_dict, cnn, I_out, I_gt, mask):
    I_comp = mask * I_out + (1.0 - mask) * I_gt
    v = valid_loss(loss_dict['valid'], I_out, I_gt, mask)
    h = hole_loss(loss_dict['hole'], I_out, I_gt, mask)
    p = perceptual_loss(loss_dict['perceptual'], cnn, I_out, I_gt, I_comp)
    s = style_loss(loss_dict['style'], cnn, I_out, I_gt, I_comp)
    tv = tv_loss(loss_dict['tv'], I_comp)
    n = I_gt.shape[0]
    return (v + h + p + s + tv) / n
           

def get_image(dataset, idx):
    # 4509, 552
    
    img, masks, mask_max = dataset[idx]
    # channel = np.min(mask_max)  # get an arbitrary foreground class  
    channel = np.max(mask_max)  # get the background class
    mask = masks[channel]
    n = 8
    img = torch.Tensor(img).unsqueeze(0).repeat(n, 1, 1, 1)
    mask = torch.Tensor(mask).unsqueeze(0).repeat(n, 1, 1, 1)
    return img, mask

def basic_loss_test():
    loss_dict = {
                'valid': 1, 
                'hole': 6,
                'perceptual': 0.05,
                'style': 120,
                'tv': 0.1,
                }
    cnn = models.vgg11(pretrained=True).features

    HEIGHT = WIDTH = 128
    val_dataset = CocoStuffDataSet(mode='val', supercategories=['animal'], height=HEIGHT, width=WIDTH)
    I_gt, mask = get_image(val_dataset, 1010)
    I_out = I_gt + 0.1 * torch.randn(I_gt.size())
    loss = total_loss(loss_dict, cnn, I_out, I_gt, mask)
    print("Loss: ", loss) 


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

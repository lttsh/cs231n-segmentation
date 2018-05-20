import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import *

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: (int) number of input channels to this block
            out_channels: (int) number of out_channels from this block
            num_conv_layers: (int) number of total Conv2d filters. Must be >= 2
        """
        super().__init__()
        self.net = nn.Sequential(
                        *(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.LeakyRelu(),
                        nn.MaxPool2d(kernel_size=2)))

    def forward(self, x):
        return self.net(x)

class GAN(nn.Module):
    def __init__(self, num_classes, masks_shape, images_shape):
        """
        Args:
            num_classes: (int) number of output classes to be predicted
            pretrained: (bool) if True loads vgg-11 pretrained weights. default=True
        """
        super().__init__()
        self.image_branch = nn.Sequential(
            *(nn.Conv2d(3, 16, kernel_size=5, padding=2),
               nn.LeakyRelu(),
               nn.Conv2d(16, 64, kernel_size=5, padding=2),
               nn.LeakyRelu())
        )
        self.masks_branch = nn.Sequential(
            *(nn.Conv2d(num_classes, 64, kernel_size=5, padding=2),
               nn.LeakyRelu())
        )

        self.enc1 = _EncoderBlock(128, 128)
        self.enc2 = nn.Sequential(
            *(nn.Conv2d(128, 128, kernel_size=3, padding=1),
               nn.LeakyRelu())
        )
        features_len = self._get_conv_output(images_shape, masks_shape)
        self.prediction = nn.Linear(features_len, 1)
        initialize_weights(self.image_branch, self.masks_branch, self.enc1, self.enc2, self.prediction)

    def forward(self, images, masks):
        """
        Args:
            images: (N, 3, H, W) input images
            masks: (N, C_classes, H, W) outputs of segmentation model or ground truth

        Return:
            (N,) tensor: vector of probabilities that images is the ground truth
                        label map of masks
        """
        features = self._forward_features(images, masks)
        prediction = self.prediction(features)
        return prediction

    # generate input sample and forward to get shape
    def _get_conv_output(self, images_shape, masks_shape):
        images = torch.rand(1, *images_shape)
        masks = torch.rand(1, *masks_shape)

        output_feat = self._forward_features(images, masks)
        return output_feat.size(1)

    def _forward_features(self, images, masks):
        images = self.image_branch(images)
        masks = self.masks_branch(masks)
        mixed = torch.cat([images, masks], 1)
        enc1 = self.enc1(mixed)
        enc2 = self.enc2(enc1)
        return flatten(enc2)

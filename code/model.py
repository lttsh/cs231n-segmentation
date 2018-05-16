import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import *

#TODO: credit https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/seg_net.py
def initialize_weights(*models):
    """
    Initializes a sequence of models
    Args:
        models: (Iterable) models to initialize. 
            each model can must be one of {nn.Conv2d, nn.Linear, nn.BatchNorm2d}
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _DecoderBlock(nn.Module):
    """
    CNN block for the decoder.
    """
    def __init__(self, in_channels, out_channels, num_conv_layers):
        """
        Args:
            in_channels: (int) number of input channels to this block
            out_channels: (int) number of out_channels from this block
            num_conv_layers: (int) number of total Conv2d filters. Must be >= 2
        """
        assert num_conv_layers >= 2
        super().__init__()
        middle_channels = in_channels // 2
        layers = [
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ]
        layers += [
                      nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
                      nn.ReLU(),
                  ] * (num_conv_layers - 2)
        layers += [
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the decoder block
        Args:
            x: (torch.Tensor) input tensor to be decoded by block
        Return:
            (torch.Tensor) result of decoding input tensor
        """
        return self.decode(x)


class VerySmallNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.net = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        return self.net(x)
        
class SegNetSmall(nn.Module):
    """
    Smaller implementation of SegNet based off of vgg-11
    """
    def __init__(self, num_classes, pretrained=True):
        """
        Args:
            num_classes: (int) number of output classes to be predicted
            pretrained: (bool) if True loads vgg-11 pretrained weights. default=True
        """
        super().__init__()
        vgg = models.vgg11(pretrained)

        features = list(vgg.features.children())
        self.enc1 = nn.Sequential(*features[:3])  # C_out = 64
        self.enc2 = nn.Sequential(*features[3:6])  # C_out = 128
        self.enc3 = nn.Sequential(*features[6:11])  # C_out = 256
        self.enc4 = nn.Sequential(*features[11:16])  # C_out = 512
        self.enc5 = nn.Sequential(*features[16:])  # C_out = 512
        for param in self.parameters():
            param.requires_grad = False

        self.dec5 = nn.Sequential(
            *([nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)] +
              [nn.Conv2d(512, 512, kernel_size=3, padding=1),
               nn.ReLU()] * 2)
        )
        self.dec4 = _DecoderBlock(1024, 256, 2)
        self.dec3 = _DecoderBlock(512, 128, 2)
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        initialize_weights(self.dec5, self.dec4, self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec5 = self.dec5(enc5)
        dec4 = self.dec4(torch.cat([enc4, dec5], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        return dec1


class SegNetSmaller(nn.Module):
    """
    Smaller implementation of SegNet based off of vgg-11
    """
    def __init__(self, num_classes, pretrained=True, freeze_pretrained=False):
        """
        Args:
            num_classes: (int) number of output classes to be predicted
            pretrained: (bool) if True loads vgg-11 pretrained weights. default=True
        """
        super().__init__()
        vgg = models.vgg11(pretrained)

        features = list(vgg.features.children())

        self.enc1 = nn.Sequential(*features[:3])  # C_out = 64
        self.enc2 = nn.Sequential(*features[3:6])  # C_out = 128
        self.enc3 = nn.Sequential(*features[6:11])  # C_out = 256
        
        if freeze_pretrained:
            for param in self.parameters():
                param.requires_grad = False

        self.dec3 = nn.Sequential(
            *([nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)] +
              [nn.Conv2d(128, 128, kernel_size=3, padding=1),
               nn.ReLU()] * 2)
        )
        self.dec2 = _DecoderBlock(256, 64, 2)
        self.dec1 = _DecoderBlock(128, num_classes, 2)
        initialize_weights(self.dec3, self.dec2, self.dec1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        dec3 = self.dec3(enc3)
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        dec1 = self.dec1(torch.cat([enc1, dec2], 1))
        return dec1


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
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2)))

    def forward(self, x):
        return self.net(x)
                    
class GAN(nn.Module):
    def __init__(self, num_classes, images_shape, masks_shape):
        """
        Args:
            num_classes: (int) number of output classes to be predicted
            pretrained: (bool) if True loads vgg-11 pretrained weights. default=True
        """
        super().__init__()
        self.image_branch = nn.Sequential(
            *(nn.Conv2d(3, 16, kernel_size=5, padding=2),
               nn.ReLU(),
               nn.Conv2d(16, 64, kernel_size=5, padding=2),
               nn.ReLU())
        )
        self.masks_branch = nn.Sequential(
            *(nn.Conv2d(num_classes, 64, kernel_size=5, padding=2),
               nn.ReLU())
        )
       
        self.enc1 = _EncoderBlock(128, 128)
        self.enc2 = nn.Sequential(
            *(nn.Conv2d(128, 128, kernel_size=3, padding=1),
               nn.ReLU())
        )
        features_len = self._get_conv_output(images_shape, masks_shape)
        self.prediction = nn.Linear(features_len, 1)
        self.probability = nn.Sigmoid()

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
        return self.probability(prediction)

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

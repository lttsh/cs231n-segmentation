import torch
import torch.nn as nn
import numpy as np

'''
    Converts a prediction Tensor (scores) into a masks
'''
def convert_to_mask(prediction):
    B, C, H, W = prediction.size()
    prediction = torch.transpose(prediction, 0, 1) # C x B x H x W
    prediction = torch.reshape(prediction, (C, -1))
    _, indices = torch.max(prediction, 0, False)
    out = torch.zeros(prediction.size())
    out[indices, np.arange(B * H * W)] = 1
    out = torch.reshape(out, (C, B, H, W))
    out = torch.transpose(out, 0, 1)
    return out # B x C x H x W where C is the number of classes


"""
Flattens input x while maintaining the batch dimension
"""
def flatten(x):
    return x.view(x.size(0), -1)

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

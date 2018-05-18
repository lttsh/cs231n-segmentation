import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

total_saved = 0
def visualize_mask(data, gt, pred):    
    num_classes = 11
    global total_saved
    for i in range(len(data)):
        total_saved += 1    
        img = data[i].detach().cpu().numpy()
        gt_mask = gt[i].detach().cpu().numpy()
        pred_mask = np.argmax(pred[i].detach().cpu().numpy(), axis=0)

        display_image = np.transpose(img, (1, 2, 0))
        plt.figure()

        plt.subplot(131)
        plt.imshow(display_image)
        plt.axis('off')
        plt.title('original image')
        
        cmap = discrete_cmap(num_classes, 'Paired')
        norm = colors.NoNorm(vmin=0, vmax=num_classes)

        plt.subplot(132)
        plt.imshow(display_image)
        plt.imshow(gt_mask, alpha=0.8, cmap=cmap, norm=norm)
        plt.axis('off')
        plt.title('real mask')

        plt.subplot(133)
        plt.imshow(display_image)
        plt.imshow(pred_mask, alpha=0.8, cmap=cmap, norm=norm)
        plt.axis('off')
        plt.title('predicted mask')
        plt.savefig('saved_{}.png'.format(total_saved))
        plt.show()


# SOurce : https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

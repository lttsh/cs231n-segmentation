import torch
import torch.nn as nn
import torchvision.transforms as T
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

COCO_ANIMAL_MEAN = [0.46942962, 0.45565367, 0.39918785]
COCO_ANIMAL_STD = [0.2529317,  0.24958833, 0.26295757]

def normalize():
    return T.Normalize(mean=COCO_ANIMAL_MEAN, std=COCO_ANIMAL_STD)

def de_normalize(images):
    """
    Normalize input batch of images

    Input:
        images: pytorch tensor with shape (3, H, W)
    Return:
        denormalized images
    """
    mean = torch.Tensor(COCO_ANIMAL_MEAN).view(-1, 1, 1)
    std = torch.Tensor(COCO_ANIMAL_STD).view(-1, 1, 1)
    return (images * std) + mean

def average_grad_norm(model):
        """
        Return the average gradient norm of the input model's parameters
        Partly copied from torch/nn/utils/clip_grad
        """

        norm_type = 2.0
        total_norm = 0
        n = 0.0
        for name, p in model.named_parameters():
            n += 1
            if p.grad is not None:
               param_norm = p.grad.data.norm(norm_type)
               total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        average_norm = total_norm / n
        return average_norm

def smooth_labels(n, device):
    """
    produces smoothed 'real' and 'fake' labels close to 1.0 and 0.0, respecitively

    Input:
    n: (int) number of real and fake labels to produce
    Return:
    false_labels: (n,1) shape Tensor of labels from 0.0 to 0.3
    true_labels: (n,1) shape Tensor of labels from 0.7 to 1.0
    """
    false_labels = 0.3 * torch.rand(n, 1).to(device)
    true_labels = 1.0 - 0.3 * torch.rand(n, 1).to(device)
    return false_labels, true_labels

# PLOT UTILS
# total_saved = 0
# def visualize_mask(data, gt, pred, save=False):
#     num_classes = 11
#     global total_saved
#     for i in range(len(data)):
#         total_saved += 1
#         img = data[i].detach().cpu().numpy()
#         gt_mask = gt[i].detach().cpu().numpy()
#         pred_mask = np.argmax(pred[i].detach().cpu().numpy(), axis=0)
#
#         display_image = np.transpose(img, (1, 2, 0))
#         plt.figure()
#
#         plt.subplot(131)
#         plt.imshow(display_image)
#         plt.axis('off')
#         plt.title('original image')
#
#         cmap = discrete_cmap(num_classes, 'Paired')
#         norm = colors.NoNorm(vmin=0, vmax=num_classes)
#
#         plt.subplot(132)
#         plt.imshow(display_image)
#         plt.imshow(gt_mask, alpha=0.8, cmap=cmap, norm=norm)
#         plt.axis('off')
#         plt.title('real mask')
#
#         plt.subplot(133)
#         plt.imshow(display_image)
#         plt.imshow(pred_mask, alpha=0.8, cmap=cmap, norm=norm)
#         plt.axis('off')
#         plt.title('predicted mask')
#         if save:
#             plt.savefig('saved_{}.png'.format(total_saved))
#         plt.show()


def visualize_mask(trainer, loader, number):
    total = 0
    to_return = []
    for data, mask_gt, gt_visual in loader:
        if total < number:
            data = data.to(trainer.device)
            batch_size = data.size()[0]
            total += batch_size
            mask_pred = convert_to_mask(trainer._gen(data))
            for i in range(len(data)):
                img = de_normalize(data[i].detach().cpu().numpy())
                gt_mask = gt_visual[i].detach().cpu().numpy()
                pred_mask = np.argmax(mask_pred[i].detach().cpu().numpy(), axis=0)
                to_return.append((img, gt_mask, pred_mask))
                display_image = np.transpose(img, (1, 2, 0))
                plt.figure()

                plt.subplot(131)
                plt.imshow(display_image)
                plt.axis('off')
                plt.title('original image')

                cmap = discrete_cmap(NUM_CLASSES, 'Paired')
                norm = colors.NoNorm(vmin=0, vmax=NUM_CLASSES)

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
                plt.show()
        else:
            break
    return to_return


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

def visualize_conf(matrix, idToCat):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    normed_conf = matrix / np.expand_dims(np.sum(matrix, axis= 1), axis=-1)
    res = ax.imshow(normed_conf, cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = matrix.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(width), idToCat)
    plt.yticks(range(height), idToCat)
    plt.show()

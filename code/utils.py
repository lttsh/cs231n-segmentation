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

def smooth_labels(n, device):
    """
    produces smoothed 'real' and 'fake' labels close to 1.0 and 0.0, respecitively

    Input:
    n: (int) number of real and fake labels to produce
    Return:
    false_labels: (n,1) shape Tensor of labels from 0.0 to factor
    true_labels: (n,1) shape Tensor of labels from 1.0-factor to 1.0
    """
    factor = 0.05
    assert factor < 0.5
    false_labels = factor * torch.rand(n, 1).to(device)
    true_labels = 1.0 - factor * torch.rand(n, 1).to(device)
    return false_labels, true_labels


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
    res = ax.imshow(normed_conf,
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

    
def true_positive_and_negative(true_scores, false_scores):
    assert true_scores.size() == false_scores.size()
    ones = torch.ones(true_scores.size())
    zeros = torch.zeros(true_scores.size())
    
    true_pos = (torch.where(true_scores > 0.5, ones, zeros)).mean()
    true_neg = 1.0 - (torch.where(false_scores > 0.5, ones, zeros)).mean()
    return true_pos, true_neg

def dominant_class(mask, numClasses):
    """
    mask: Tensor (B, H, W)
    Return:
        numpy array (B,) - dominant non-background class of each image
        
    """
    mask = flatten(mask).cpu().numpy()
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=numClasses), axis=1, arr=mask)
    counts = counts[:, :-1]  # get counts, ignoring background class
    return np.argmax(counts, axis=1)


"""
Evaluation functions
"""
def calc_pixel_accuracy(labels, preds, state):
    if state is None:
        state = {
            'true_pos' : 0.0,
            'total_pix' : 0.0,
        }
    state['true_pos'] += torch.sum(preds * labels).item()
    state['total_pix'] += torch.sum(labels).item()
    state['final'] = float(state['true_pos']) / (state['total_pix'] + 1e-12)
    return state

def calc_mean_IoU(labels, preds, state):
    if state is None:
        state = {
            'total' : 0.0,
            'mIoU' : 0.0,
        }
    
    batch_size, num_classes = labels.size()[:2]
    state['total'] += batch_size
    labels = labels.view(batch_size, num_classes, -1)
    preds = preds.view(batch_size, num_classes, -1)
    total_pix = torch.sum(labels, 2)
    class_present = (total_pix > 0).float() # Ignore class that was not originally present in the groundtruth
    true_positive = torch.sum(labels * preds, 2)
    false_positive = torch.sum(preds, 2) - true_positive
    numerator = torch.sum(class_present * (true_positive / (total_pix + false_positive + 1e-12)), 1)
    denominator = class_present.sum(1)
    fraction = (numerator / denominator).masked_select(denominator > 0)
    state['mIoU'] +=  torch.sum(fraction).item()
    state['final'] = state['mIoU'] / state['total']
    return state

def per_class_pixel_acc(labels, preds, state):
    ''' Evaluates per class pixel accuracy for given batch
    labels: Bx CxHxW
    preds: Bx CxHxW
    state: dictionary containing
        'true_pos': Cx1 numpy array that totals the number of true positives per class
        'total_pix': Cx1 numpy array that totals the number of pixels per class
    '''
    numClasses = labels.size()[1]
    if state is None:
        state = {
            'true_pos': np.zeros(numClasses),
            'total_pix': np.zeros(numClasses),
        }

    positives = preds * labels # B x C x H x W
    positives = positives.transpose(0, 1).contiguous().view(numClasses, -1).cpu().numpy() # C x -1
    state['true_pos'] += np.sum(positives, 1)
    allexamples = labels.transpose(0, 1).contiguous().view(numClasses, -1).cpu().numpy()
    state['total_pix'] += np.sum(allexamples, 1)
    state['final'] = np.mean(state['true_pos'] / (state['total_pix'] + 1e-12))
    return state

def Conv2d_BatchNorm2d(in_channels, out_channels, kernel_size, padding, use_bn):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
    if use_bn:
        layers += [nn.BatchNorm2d(out_channels)]
    return layers


'''
Save mask to file
'''
def save_to_file(pred_mask, display_image, gt_mask, i, save_dir):
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
    plt.savefig(os.path.join(save_dir, str(i) + '.png'))
    plt.clf()
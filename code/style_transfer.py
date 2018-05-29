import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import PIL
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CocoStuffDataSet

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).
    
    Returns:
    - scalar content loss
    """
    _, C_l, H_l, W_l = content_current.size()
    cc = content_current.view(C_l, H_l*W_l)
    ct = content_original.view(C_l, H_l*W_l)
    return content_weight * (cc-ct).pow(2).sum()

def gram_matrix(features, feature_mask=None, normalize=True):
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
    F_0 = F_1 = features.view(N, C, -1)

    if feature_mask is not None:
        T = feature_mask.view(*feature_mask.shape[:-2], -1)
        # print("F shape: ", F_1.shape)
        # print("T shape: ", T.shape)
        # print("Feature mask shape: ", feature_mask.shape)
        F_1 = F_1 * T
    G = torch.matmul(F_0, F_1.transpose(1, 2))
    if normalize:
        G /= (C*H*W)
    return G

def style_loss(feats, style_layers, style_targets, style_weights, feature_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """

    loss = 0

    for i in range(len(style_layers)):
        if feature_masks is not None:
            G = gram_matrix(feats[style_layers[i]], feature_masks[style_layers[i]])
        else:
            G = gram_matrix(feats[style_layers[i]])
        loss += style_weights[i] * (style_targets[i] - G).pow(2).sum()
    return loss

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    N, C, H, W = img.size()
    down = torch.cat((img[:,:,1:,:], img[:,:,-1,:].view(N, C, 1, W)), dim=2)
    right = torch.cat((img[:,:,:,1:], img[:,:,:,-1].view(N, C, H, 1)), dim=3)
    return tv_weight * ((down - img).pow(2).sum() + (right - img).pow(2).sum())

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.
    
    Inputs:
    - x: A PyTorch Tensor of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.
    
    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Tensor of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def style_transfer(content_image, style_image, content_mask, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, savename, init_random=False, mask_layer=False):
    """
    Run style transfer!
    
    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    

    # Extract features for the content image
    content_img = preprocess(content_image, size=image_size)
    feats = extract_features(content_img, cnn)
    content_target = feats[content_layer].clone()

    # Extract features for the style image
    style_img = preprocess(style_image, size=style_size)
    feats = extract_features(style_img, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or noise
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img.requires_grad_()
    
    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img Torch tensor, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img], lr=initial_lr)
    
    plt.figure()
    
    feature_masks = None
    if mask_layer:
        feature_masks = get_soft_masks(content_mask, cnn, style_layers)
        # for m in feature_masks:
        #     m = m.detach().numpy().reshape(*(m.shape)[-2:])
        #     plt.axis('off')
        #     plt.imshow(m)
        #     plt.show()
    for t in range(200):
        if t < 190:
            img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(img, cnn)
        
        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights, feature_masks)
        t_loss = tv_loss(img, tv_weight)
        loss = c_loss + s_loss + t_loss
        
        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img], lr=decayed_lr)
        optimizer.step()

        if t % 10 == 0:
            print("Iteration {}".format(t))

    img = np.asarray(deprocess(img.data.cpu()), dtype=np.uint8)
    content_img = np.asarray(deprocess(content_img), dtype=np.uint8)
    content_mask = np.expand_dims(content_mask, axis=-1)
    final_img = img.astype(np.uint8)
    final_img = PIL.Image.fromarray(final_img)
    plt.axis('off')
    plt.imshow(final_img)
    plt.savefig(savename)
    plt.show()

# The setup functions
SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def get_image(dataset, idx):
    # 4509, 552
    
    img, masks, mask_max = dataset[idx]
    # channel = np.min(mask_max)  # get an arbitrary foreground class  
    channel = np.max(mask_max)  # get the background class
    img = PIL.Image.fromarray(np.uint8(img.numpy().transpose(1, 2, 0)*255.))
    mask = masks[channel]
    return img, mask

def corresponding_filter(layer):
    if isinstance(layer, nn.Conv2d):
        return nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
    elif isinstance(layer, nn.MaxPool2d):
        return layer
    else:
        return None

def get_soft_masks(mask, cnn, layer_indices):
    x = torch.Tensor(mask.reshape((1, 1, *mask.shape)))
    soft_masks = []
    for i, module in enumerate(cnn._modules.values()):
        next_module = corresponding_filter(module)
        if next_module:
            x = next_module(x)
            x.requires_grad = False
        if i in layer_indices:
            soft_masks.append(x)
        else:
            soft_masks.append(None)
    return soft_masks


if __name__ == "__main__":
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    HEIGHT = WIDTH = 256
    val_dataset = CocoStuffDataSet(mode='val', supercategories=['animal'], height=HEIGHT, width=WIDTH)
    content_image, content_mask = get_image(val_dataset, 1010)

    # cnn = torchvision.models.squeezenet1_1(pretrained=True).features
    # style_layers = (1, 4, 6, 7)

    cnn = torchvision.models.vgg11(pretrained=True).features
    style_layers = (3, 8, 13, 18)

    cnn.type(dtype)
    # We don't want to train the model any further, so we don't want PyTorch to waste computation 
    # computing gradients on parameters we're never going to update.
    for param in cnn.parameters():
        param.requires_grad = False

    style_dir = '../styles/'
    for style_image_name in os.listdir(style_dir):
        if 'starry' not in style_image_name:
            continue
        style_image = PIL.Image.open(os.path.join(style_dir, style_image_name))
        transfer_params = {
            'content_image' : content_image,
            'style_image' : style_image,
            'content_mask': content_mask,
            'image_size' : HEIGHT,
            'style_size' : HEIGHT,
            'content_layer' : 6,
            'content_weight' : 1e-3, 
            'style_layers' : style_layers,
            'style_weights' : (20000, 500, 12, 1),
            'tv_weight' : 1e-2,
            'savename' : style_image_name,
            'init_random' : False,
            'mask_layer' : True,
        }

        style_transfer(**transfer_params)

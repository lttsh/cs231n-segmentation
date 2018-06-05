import os, argparse
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor   

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
    cc = content_current.view(C_l, H_l*W_l).to(device)
    ct = content_original.view(C_l, H_l*W_l).to(device)
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
    F_0 = F_1 = features.view(N, C, -1).to(device)

    if feature_mask is not None:
        T = feature_mask.view(*feature_mask.shape[:-2], -1).to(device)
#         print("F shape: ", F_1.shape)
#         print("T shape: ", T.shape)
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
        loss += style_weights[i] * (style_targets[i].to(device) - G).pow(2).sum()
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
    down = torch.cat((img[:,:,1:,:], img[:,:,-1,:].view(N, C, 1, W)), dim=2).to(device)
    right = torch.cat((img[:,:,:,1:], img[:,:,:,-1].view(N, C, H, 1)), dim=3).to(device)
    img = img.to(device)
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
    prev_feat = x.to(device)
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

# Extract features for the style image
def prep_style(cnn, style_img, style_size, style_layers):
    style_img = preprocess(style_img, size=style_size)
    feats = extract_features(style_img, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))
    return style_img, style_targets

def style_transfer(cnn, content_image, style_image, content_mask, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, max_iters, init_random=False, mask_layer=False, second_style_image=None):
    """
    Run style transfer!
    
    Inputs:
    - cnn: cnn model 
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image 
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - max_iters: number of iterations to run 
    - init_random: initialize the starting image to uniform random noise
    - mask_layer: (bool) if True, use masking on gram matrices.
    - second_style_image: second style image to use on the foreground of image
    """
    # Extract features for the content image
    content_img = preprocess(content_image, size=image_size).to(device)
    feats = extract_features(content_img, cnn)
    content_target = feats[content_layer].clone().to(device)

    style_image, style_targets = prep_style(cnn, style_image, style_size, style_layers)
    if second_style_image is not None:
        second_style_image, second_style_targets = prep_style(cnn, second_style_image, style_size, style_layers)

    # Initialize output image to content image or noise
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1).type(dtype)
    else:
        img = content_img.clone().type(dtype)

    img = img.to(device)
    # We do want the gradient computed on our image!
    img.requires_grad_()
    
    # Set up optimization hyperparameters
    initial_lr = 3e-2

    # Note that we are optimizing the pixel values of the image by passing
    # in the img Torch tensor, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img], lr=initial_lr)
        
    feature_masks = None
    if mask_layer:
        feature_masks = get_soft_masks(content_mask, cnn, style_layers, style_size)
        # for m in feature_masks:
        #     m = m.detach().numpy().reshape(*(m.shape)[-2:])
        #     plt.axis('off')
        #     plt.imshow(m)
        #     plt.show()
        second_feature_masks = get_soft_masks(1.0 - content_mask, cnn, style_layers, style_size)
    loss_list = []
    for t in range(max_iters):
#         if t < 390:
#             img.data.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(img, cnn)
        
        # Compute loss
        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights, feature_masks)
        if second_style_image is not None:
            second_style_weights = [x / 10 for x in style_weights]
            second_s_loss = style_loss(feats, style_layers, second_style_targets, second_style_weights, second_feature_masks)
            s_loss += second_s_loss
        t_loss = tv_loss(img, tv_weight)
        loss = c_loss + s_loss + t_loss
        loss_list.append(loss.item())
        loss.backward()

        # Perform gradient descents on our image values
#         if t == decay_lr_at:
#             optimizer = torch.optim.Adam([img], lr=decayed_lr)
        optimizer.step()

        if t % 100 == 0:
            print("Iteration {},\tLoss: {},\tContent: {},\tStyle: {},\tTV: {}".format(t, loss, c_loss, s_loss, t_loss))

    img = np.asarray(deprocess(img.data.cpu()), dtype=np.uint8)
#     content_img = np.asarray(deprocess(content_img.cpu()), dtype=np.uint8)
#     content_mask = np.expand_dims(content_mask, axis=-1)
    final_img = img.astype(np.uint8)
    final_img = PIL.Image.fromarray(final_img)
    return final_img, loss, loss_list

# The setup functions
# SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COCO_ANIMAL_MEAN = np.array([0.46942962, 0.45565367, 0.39918785], dtype=np.float32)
COCO_ANIMAL_STD = np.array([0.2529317,  0.24958833, 0.26295757], dtype=np.float32)

def preprocess(img, size=512):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=COCO_ANIMAL_MEAN.tolist(),
                    std=COCO_ANIMAL_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in COCO_ANIMAL_STD.tolist()]),
        T.Normalize(mean=[-m for m in COCO_ANIMAL_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def get_image_from_dataset(dataset, idx):    
    img, masks, mask_max = dataset[idx]
    channel = np.max(mask_max)  # get the background class (or if no background, class with highest index)
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

def get_soft_masks(mask, cnn, layer_indices, style_size):
    #first reshape the mask to be (style_size, style_size) in shape
    x = torch.Tensor(mask.reshape((1, 1, *mask.size())))
#     x = nn.functional.upsample(x, size=(style_size, style_size))
    
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

def display_style_transfer(img, savename):
    plt.figure()
    plt.axis('off')
#     img = PIL.ImageEnhance.Contrast(img).enhance(1.3)
#     img = PIL.ImageEnhance.Brightness(img).enhance(1.1)
    plt.imshow(img)
    plt.savefig(savename)
    plt.show()

if __name__ == "__main__":
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('-b' , '--background_style', default='starry_night.jpg', type=str,
                        help='filename for the background style')
    parser.add_argument('-f' , '--foreground_style', default=None, type=str,
                        help='filename for the foreground style')
    parser.add_argument('-s', '--im_size', default=256, type=int,
                        help='desired size of input images')
    parser.add_argument('-i', '--content_index', default=417, type=int,
                        help='index of context image in coco dataset')
    # suggested indices: 417, 77, 1011, 913, 55
    args = parser.parse_args()

    HEIGHT = WIDTH = args.im_size
    val_dataset = CocoStuffDataSet(mode='val', supercategories=['animal'], height=HEIGHT, width=WIDTH, do_normalize=False)
    content_image, background_mask = get_image(val_dataset, args.content_index)
    foreground_mask = 1.0 - background_mask

    cnn = torchvision.models.vgg16(pretrained=True).features
    style_layers = (0, 5, 10, 17, 24)
    style_weights = np.ones(5).tolist()
    cnn.type(dtype)
    # We don't want to train the model any further, so we don't want PyTorch to waste computation 
    # computing gradients on parameters we're never going to update.
    for param in cnn.parameters():
        param.requires_grad = False

    style_dir = '../styles/'
    style_background_name = args.background_style
    style_foreground_name = args.foreground_style
    
    style_background_image = PIL.Image.open(os.path.join(style_dir, style_background_name))
    if style_foreground_name:
        style_foreground_image = PIL.Image.open(os.path.join(style_dir, style_foreground_name))
    else:
        style_foreground_image = None

    transfer_params = {
        'cnn' : cnn,
        'content_image' : content_image,
        'style_image' : style_background_image,
        'content_mask': background_mask,
        'image_size' : HEIGHT,
        'content_layer' : 12,
        'content_weight' : 1e-3,
        'style_layers' : style_layers,
        'style_weights' : style_weights,
        # 'tv_weight' : 1e-2,
        'tv_weight' : 0,
        'init_random' : False,
        'mask_layer' : True,
        'second_style_image' : style_foreground_image 
    }

    final_img = style_transfer(**transfer_params)
    display_style_transfer(final_img, 'test.png')
    
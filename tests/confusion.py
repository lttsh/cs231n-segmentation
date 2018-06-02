import numpy as np
import torch

from code.utils import visualize_conf

def get_confusion_matrix(pred, gt, numClasses):
    ''' Method to get confusion matrix, returns C x C numpy array
    pred: B x C x H x W masks (0 / 1)
    gt: B x H x W ground truth labels
    '''
    mask_pred = np.transpose(pred.numpy(), axes=(1, 0, 2, 3)) # C x B x H x W
    pred_labels = np.argmax(mask_pred, axis=0).reshape((-1,))
    gt_labels = gt.numpy().reshape((-1,))
    print (pred_labels, gt_labels)
    confusion_mat = np.zeros((numClasses, numClasses))
    x = pred_labels + numClasses * gt_labels
    bincount_2d = np.bincount(x.astype(np.int32),
                          minlength=numClasses ** 2)
    assert bincount_2d.size == numClasses ** 2
    conf = bincount_2d.reshape((numClasses, numClasses))
    return conf

pred = np.array([[
    [[1, 0], [0, 0]],
    [[0, 1], [0, 1]],
    [[0, 0], [1, 0]]
    ]])
gt = np.array([
    [[1, 0], [2, 0]]
])
print (pred.shape) # 1 x 3 x 2 x 2
print (gt.shape) # 1 x 2 x 2

pred_tensor = torch.from_numpy(pred)
gt_tensor = torch.from_numpy(gt)

confusion = get_confusion_matrix(pred_tensor, gt_tensor, 3)
print (confusion)

visualize_conf(confusion, ['cat', 'dog', 'background'])

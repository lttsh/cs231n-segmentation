import numpy as np
import torch

pred = np.array([[
    [[1, 0], [0, 0]],
    [[0, 1], [0, 1]],
    [[0, 0], [1, 0]]
    ]])
gt = np.array([[
    [[1, 0], [0, 0]],
    [[0, 1], [1, 0]],
    [[0, 0], [0, 1]]
    ]])
tensor_pred = torch.from_numpy(pred)
tensor_gt = torch.from_numpy(gt)

print (tensor_pred.size(), tensor_gt.size())

true_pos = np.zeros(3)
total_pix = np.zeros(3)

positives = tensor_pred * tensor_gt # B x C x H x W
positives = positives.transpose(0, 1).view((3, -1)) # C x -1
print (positives)
true_pos += torch.sum(positives, 1).numpy()
print (true_pos)
allexamples = tensor_gt.transpose(0, 1).view((3, -1))
total_pix += torch.sum(allexamples, 1).numpy()

print (total_pix)

print (true_pos / total_pix)

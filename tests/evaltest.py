import torch
import numpy as np


prediction = np.zeros((2, 2, 2, 2))
prediction[0, 1, :, :] = 2
prediction[0, 0, :1, :1] = 5
prediction[1, 1, 1:, 1:] = 2
prediction[1, 0, :, :] = 0
tensor_pred = torch.from_numpy(prediction)
print(tensor_pred.size()[0])
groundtruth = prediction
tensor_gt = torch.from_numpy(groundtruth)

mask_gt = convert_to_mask(tensor_gt)
mask_pred = convert_to_mask(tensor_pred)
mask_gt[1] = mask_gt[0]
print (mask_gt)
print(mask_pred)
""" Test pixel accuracy """

true_pos = 0
total_pix = 0
for i in range(2):
    true_pos += torch.sum(mask_gt[i] * mask_pred[i]).item()
    total_pix += torch.sum(mask_gt[i]).item()

print (true_pos, total_pix)

''' Test mean IOU '''
numClasses = 2
batch_size = 2
truepos = torch.zeros(batch_size, numClasses) # Number of pixels with correct label
falsepos = torch.zeros(batch_size, numClasses) # Number of pixels from class i incorrectly labeled
totalpix = torch.zeros(batch_size, numClasses) # Number of pixels in ground truth with label i

mask_gt = mask_gt.reshape((batch_size, numClasses, -1))
mask_pred = mask_pred.reshape((batch_size, numClasses, -1))
truepositive = torch.sum(mask_gt * mask_pred, 2)
totalpix += torch.sum(mask_gt, 2)
truepos += truepositive
falsepos += torch.sum(mask_pred, 2) - truepositive
mIOU  = 1.0 / numClasses * torch.sum(truepos / (totalpix + falsepos), 1)
print (mIOU)

numClasses = 2
total = 0
mIOU = 0.0
for i in range(2):
    batch_size = 1
    total += batch_size
    mask_pred_b = mask_pred[i]
    mask_gt_b = mask_gt[i].reshape((batch_size, numClasses, -1))
    mask_pred_b = mask_pred_b.reshape((batch_size, numClasses, -1))
    truepositive = torch.sum(mask_gt_b * mask_pred_b, 2)
    totalpix = torch.sum(mask_gt_b, 2)
    truepos = truepositive
    falsepos = torch.sum(mask_pred_b, 2) - truepositive
    mIOU += 1.0 / numClasses * torch.sum(truepos / (totalpix + falsepos))
print(1.0 / total * mIOU)

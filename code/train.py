import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CocoStuffDataSet

import numpy as np
# train_loader = CocoStuffDataSet(train_set, batch_size=32, shuffle=True)
# test_loader = CocoStuffDataSet(test_set, batch_size=32, shuffle=False)

def convert_to_mask(prediction):
    B, C, H, W = prediction.size()
    prediction = torch.transpose(prediction, 0, 1) # C x B x H x W
    prediction = torch.reshape(prediction, (C, -1))
    _, indices = torch.max(prediction, 0, False)
    out = torch.zeros(prediction.size())
    out[indices, np.arange(B * H * W)] = 1
    out = torch.reshape(out, (C, B, H, W))
    out = torch.transpose(out, 0, 1)
    return out

class Trainer():
    def __init__(self, net, train_loader, val_loader):
        self._net = net
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._net.parameters())

    def _train_batch(self, mini_batch_data, mini_batch_labels):
        self._optimizer.zero_grad()
        out = self._net(mini_batch_data)
        loss = self._criterion(out, mini_batch_labels)
        loss.backward()
        self._optimizer.step()

        return loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print ("Starting epoch {}".format(epoch))
            for mini_batch_data, mini_batch_labels in self._train_loader:
                self._train_batch(mini_batch_data, mini_batch_labels)

            # for batch_data, batch_labels in sef._val_loader:

    '''
    Converts prediction from network (B x C x H x W) into a one-hot encoded mask
    '''
    def convert_to_mask(self, prediction):
        B, C, H, W = tensor_pred.size()
        tensor_pred = torch.transpose(tensor_pred, 0, 1) # C x B x H x W
        tensor_pred = torch.reshape(tensor_pred, (C, -1))
        _, indices = torch.max(tensor_pred, 0, False)
        out = torch.zeros(tensor_pred.size())
        out[indices, np.arange(B * H * W)] = 1
        out = torch.reshape(out, (C, B, H, W))
        out = torch.transpose(out, 0, 1)
        return out


    '''
    Evaluation methods
    '''
    def evaluate_pixel_accuracy(self, loader):
        true_pos = 0
        total_pix = 0
        for mini_batch_data, mini_batch_labels in loader:
            mini_batch_prediction = self._net(mini_batch_data)
            mini_batch_prediction = convert_to_mask(mini_batch_prediction)
            ## This assumes mini_batch_pred and mini_batch labels are of size B x C x H x W
            true_pos += torch.sum(mini_batch_prediction * mini_batch_labels).item()
            total_pix += torch.sum(mini_batch_labels).item()
        return float(true_pos) / total_pix

    def evaluate_pixel_mean_acc(self, loader):
        pix_acc = self.evaluate_pixel_accuracy(loader)
        return 1.0 / loader.numClasses * pix_acc

    def evaluate_meanIOU(self, loader):
        numClasses = loader.numClasses
        total = 0
        mIOU = 0.0
        for data, mask_gt in loader:
            batch_size = data.size()[0]
            total += batch_size
            mask_pred = convert_to_mask(self._net(data))
            truepos = torch.zeros(batch_size, numClasses) # Number of pixels with correct label
            falsepos = torch.zeros(batch_size, numClasses) # Number of pixels from class i incorrectly labeled
            totalpix = torch.zeros(batch_size, numClasses) # Number of pixels in ground truth with label i
            mask_gt = mask_gt.reshape((batch_size, numClasses, -1))
            mask_pred = mask_pred.reshape((batch_size, numClasses, -1))
            truepositive = torch.sum(mask_gt * mask_pred, 2)
            totalpix += torch.sum(mask_gt, 2)
            truepos += truepositive
            falsepos += torch.sum(mask_pred, 2) - truepositive
            mIOU += 1.0 / numClasses * torch.sum(truepos / (totalpix + falsepos))
        return 1.0 / total * mIOU

if __name__ == "__main__":
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
        truepos = torch.zeros(batch_size, numClasses) # Number of pixels with correct label
        falsepos = torch.zeros(batch_size, numClasses) # Number of pixels from class i incorrectly labeled
        totalpix = torch.zeros(batch_size, numClasses) # Number of pixels in ground truth with label i
        mask_gt_b = mask_gt[i].reshape((batch_size, numClasses, -1))
        mask_pred_b = mask_pred_b.reshape((batch_size, numClasses, -1))
        truepositive = torch.sum(mask_gt_b * mask_pred_b, 2)
        totalpix += torch.sum(mask_gt_b, 2)
        truepos += truepositive
        falsepos += torch.sum(mask_pred_b, 2) - truepositive
        mIOU += 1.0 / numClasses * torch.sum(truepos / (totalpix + falsepos))
    print(1.0 / total * mIOU)

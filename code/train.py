import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CocoStuffDataSet
from model import SegNetSmall
import numpy as np

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
        """
        Training class for a specified model
        Args:
            net: (model) model to train
            train_loader: (DataLoader) train data
            val_load: (DataLoader) validation data
        """
        self._net = net
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._net.parameters())

    def _train_batch(self, mini_batch_data, mini_batch_labels):
        """
        Performs one gradient step on a minibatch of data
        Args:
            mini_batch_data: (torch.Tensor) shape (N, C_in, H, W)
                where self._net operates on (C_in, H, W) dimensional images
            mini_batch_labels: (torch.Tensor) shape (N, C_out, H, W)
                a batch of (H, W) binary masks for each of C_out classes
        Return:
            loss: (float) loss computed by self._criterion on input minibatch
        """
        self._optimizer.zero_grad()
        out = self._net(mini_batch_data)
        loss = self._criterion(out, mini_batch_labels)
        loss.backward()
        self._optimizer.step()

        return loss

    #TODO: Add validation accuracy metric
    def train(self, num_epochs, print_every=100):
        """
        Trains the model for a specified number of epochs
        Args:
            num_epochs: (int) number of epochs to train
            print_every: (int) number of minibatches to process before
                printing loss. default=100
        """
        print_i = 0
        for epoch in range(num_epochs):
            print ("Starting epoch {}".format(epoch))
            for mini_batch_data, mini_batch_labels in self._train_loader:
                loss = self._train_batch(mini_batch_data, mini_batch_labels)
                if print_i % print_every == 0:
                    print("Loss: {}".format(loss))
                print_i += 1

            # for batch_data, batch_labels in sef._val_loader:

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

    num_classes = 11
    batch_size = 1
    net = SegNetSmall(num_classes, pretrained=True)
    train_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='train'), batch_size, shuffle=True)
    val_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='val'), batch_size, shuffle=False)

    trainer = Trainer(net, train_loader, val_loader)

    trainer.train(num_epochs=5, print_every=10)

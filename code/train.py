import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CocoStuffDataSet
from model import SegNetSmall, VerySmallNet
import numpy as np

from utils import *

class Trainer():
    def __init__(self, net, train_loader, val_loader):
        """
        Training class for a specified model
        Args:
            net: (model) model to train
            train_loader: (DataLoader) train data
            val_load: (DataLoader) validation data
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print ("Using device %s" % self.device)
        self._net = net.to(self.device)
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
        mini_batch_data = mini_batch_data.to(self.device)
        mini_batch_labels = mini_batch_labels.to(self.device)
        out = self._net(mini_batch_data)
        loss = self._criterion(out, mini_batch_labels)
        loss.backward()
        self._optimizer.step()

        return loss

    def train(self, num_epochs, print_every=100, eval_every=500, eval_debug=False):
        """
        Trains the model for a specified number of epochs
        Args:
            num_epochs: (int) number of epochs to train
            print_every: (int) number of minibatches to process before
                printing loss. default=100
        """
        iter = 0
        for epoch in range(num_epochs):
            print ("Starting epoch {}".format(epoch))
            for mini_batch_data, _ , mini_batch_labels in self._train_loader:
                loss = self._train_batch(mini_batch_data, mini_batch_labels)
                if iter % print_every == 0:
                    print("Loss at iteration {}: {}".format(iter, loss))
                if iter % eval_every == 0:
                    mIOU = self.evaluate_meanIOU(self._val_loader, eval_debug)
                    print ("Mean IOU at iteration {} : {}".format(iter, mIOU))
                iter += 1

    '''
    Evaluation methods
    '''
    def evaluate_pixel_accuracy(self, loader):
        true_pos = 0
        total_pix = 0
        for mini_batch_data, mini_batch_labels, _ in loader:
            mini_batch_prediction = self._net(mini_batch_data)
            mini_batch_prediction = convert_to_mask(mini_batch_prediction)
            ## This assumes mini_batch_pred and mini_batch labels are of size B x C x H x W
            true_pos += torch.sum(mini_batch_prediction * mini_batch_labels).item()
            total_pix += torch.sum(mini_batch_labels).item()
        return float(true_pos) / total_pix

    def evaluate_pixel_mean_acc(self, loader):
        pix_acc = self.evaluate_pixel_accuracy(loader)
        return 1.0 / loader.dataset.numClasses * pix_acc

    def evaluate_meanIOU(self, loader, debug=False):
        numClasses = loader.dataset.numClasses
        total = 0
        mIOU = 0.0
        iter = 0
        for data, mask_gt, _ in loader:
            data = data.to(self.device)
            batch_size = data.size()[0]
            total += batch_size
            mask_pred = convert_to_mask(self._net(data))
            mask_gt = mask_gt.view((batch_size, numClasses, -1)).type(dtype=torch.float32).to(self.device)
            mask_pred = mask_pred.view((batch_size, numClasses, -1)).to(self.device)

            totalpix = torch.sum(mask_gt, 2)
            classPresent = (totalpix > 0).type(dtype=torch.float32) # Ignore class that was not originally present in the groundtruth
            truepositive = torch.sum(mask_gt * mask_pred, 2)
            falsepos = torch.sum(mask_pred, 2) - truepositive
            mIOU += 1.0 / numClasses * torch.sum(classPresent * (truepositive / (totalpix + falsepos + 1e-6))).item()
            iter += 1
            if debug:
                print ("Processed %d batches out of %d, accumulated mIOU : %f" % (iter, len(loader), mIOU))
        return 1.0 / total * mIOU

if __name__ == "__main__":
    num_classes = 11
    batch_size = 8
    # net = SegNetSmall(num_classes, pretrained=True)
    # net = VerySmallNet(num_classes)
    net = SegNetSmaller(num_classes, pretrained=True)
    train_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='train'), batch_size, shuffle=True)
    val_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='val'), batch_size, shuffle=False)

    trainer = Trainer(net, train_loader, val_loader)

    trainer.train(num_epochs=5, print_every=10)

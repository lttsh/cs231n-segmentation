import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CocoStuffDataSet
from model import SegNetSmall


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
            print("Starting epoch {}".format(epoch))
            for mini_batch_data, mini_batch_labels in self._train_loader:
                loss = self._train_batch(mini_batch_data, mini_batch_labels)
                if print_i % print_every == 0:
                    print("Loss: {}".format(loss))
                print_i += 1

            # for batch_data, batch_labels in sef._val_loader:

if __name__ == '__main__':
    num_classes = 10
    batch_size = 1
    net = SegNetSmall(num_classes, pretrained=True)
    train_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='train'), batch_size, shuffle=True)
    val_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='val'), batch_size, shuffle=False)
    
    trainer = Trainer(net, train_loader, val_loader)

    trainer.train(num_epochs=5, print_every=10)




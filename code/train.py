import torch
import torch.nn as nn
import torch.optim as optim
import CocoStuffDataSet

train_loader = CocoStuffDataSet(train_set, batch_size=32, shuffle=True)
test_loader = CocoStuffDataSet(test_set, batch_size=32, shuffle=False)

class Trainer():
    def __init__(self, net, train_loader, val_loader):
        self._net = net
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._net.parameters())

    def _train_batch(self, mini_batch_data, mini_batch_labels):
        self._optimizer.zero_grad()
        out = self._net(batch_data)
        loss = self._criterion(out, batch_labels)
        loss.backward()
        self._optimizer.step()

        return loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print "Starting epoch {}".format(epoch)
            for mini_batch_data, mini_batch_labels in self._train_loader:
                self._train_batch(mini_batch_data, mini_batch_labels)

            for batch_data, batch_labels in sef._val_loader






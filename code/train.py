import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
import os
import shutil
from utils import *
from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, generator, discriminator, train_loader, val_loader, \
            gan_reg=0.1, d_iters=5,  save_path=None, best_path=None, resume=False):
        """
        Training class for a specified model
        Args:
            net: (model) model to train
            train_loader: (DataLoader) train data
            val_load: (DataLoader) validation data
            gan_reg: Hyperparameter for the GAN loss (\lambda in the paper)
            save_path: path to last saved checkpoint
            best_path: path to best saved checkpoint
            resume: load from last saved checkpoint ?
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print ("Using device %s" % self.device)
        self._gen = generator.to(self.device)

        if discriminator is not None:
            self._disc = discriminator.to(self.device)
            self._discoptimizer = optim.Adam(self._disc.parameters()) # Discriminator optimizer (needs to be separate)
            self._BCEcriterion = nn.BCELoss() # Criterion for GAN loss
        else:
            print ("Runing network without GAN loss.")
            self._disc = None
            self._discoptimizer = None
            self._BCEcriterion = None

        self._train_loader = train_loader
        self._val_loader = val_loader

        self._MCEcriterion = nn.CrossEntropyLoss() # Criterion for segmentation loss
        self._genoptimizer = optim.Adam(self._gen.parameters()) # Generator optimizer

        self.gan_reg = gan_reg
        self.d_iters = d_iters
        self.start_epoch = 0
        self.best_mIOU = 0
        self.save_path = save_path
        self.best_path = best_path
        if resume:
            self.load_model()

    def _train_batch(self, mini_batch_data, mini_batch_labels, mini_batch_labels_flat):
        """
        Performs one gradient step on a minibatch of data
        Args:
            mini_batch_data: (torch.Tensor) shape (N, C_in, H, W)
                where self._gen operates on (C_in, H, W) dimensional images
            mini_batch_labels: (torch.Tensor) shape (N, C_out, H, W)
                a batch of (H, W) binary masks for each of C_out classes
            mini_batch_labels_flat: (torch.Tensor) shape (N, H, W)
                a batch of (H, W) binary masks for each of C_out classes
        Return:
            d_loss: (float) discriminator loss
            g_loss: (float) generator loss
            segmentation_loss: (float) segmentation loss
        """
        self._genoptimizer.zero_grad()
        mini_batch_data = mini_batch_data.to(self.device) # Input image (B, 3, H, W)
        mini_batch_labels = mini_batch_labels.to(self.device) # Ground truth mask (B, C, H, W)
        mini_batch_labels_flat = mini_batch_labels_flat.to(self.device) # Groun truth mask flattened (B, H, W)
        gen_out = self._gen(mini_batch_data).to(self.device) # Segmentation output from generator (B, C, H , W)
        gan_labels = torch.ones(1).to(self.device)
        g_loss = 0
        d_loss = 0

        # Minimize GAN Loss
        if self._disc is not None:
            self._discoptimizer.zero_grad()
            for i in range(self.d_iters):
                scores_false = self._disc(mini_batch_data, gen_out) # (B,)
                scores_true = self._disc(mini_batch_data, mini_batch_labels) # (B,)
                d_loss = self._BCEcriterion(scores_true, gan_labels) - self._BCEcriterion(scores_false, gan_labels)
                d_loss.backward()
                self._discoptimizer.step()
            scores_false = self._disc(mini_batch_data, gen_out)
            g_loss = self._BCEcriterion(scores_false, gan_labels)

        # Minimize segmentation loss
        segmentation_loss = self._MCEcriterion(gen_out, mini_batch_labels_flat)
        gen_loss = segmentation_loss + self.gan_reg * g_loss
        gen_loss.backward()
        self._genoptimizer.step()
        return d_loss, g_loss, segmentation_loss

    def train(self, num_epochs, print_every=100, eval_every=500, eval_debug=False):
        """
        Trains the model for a specified number of epochs
        Args:
            num_epochs: (int) number of epochs to train
            print_every: (int) number of minibatches to process before
                printing loss. default=100
        """
        writer = SummaryWriter()

        iter = 0
        batch_size = self._train_loader.batch_size
        num_samples = len(self._train_loader.dataset)
        epoch_len = int(num_samples / batch_size)

        for epoch in range(self.start_epoch, num_epochs):
            print ("Starting epoch {}".format(epoch))
            for mini_batch_data, mini_batch_labels, mini_batch_labels_flat in self._train_loader:
                self._gen.train()
                if self._disc is not None:
                    self._disc.train()
                d_loss, g_loss, segmentation_loss = self._train_batch(mini_batch_data, mini_batch_labels, mini_batch_labels_flat)
                if iter % print_every == 0:
                    writer.add_scalar('Train/SegmentationLoss', segmentation_loss, iter)
                    if self._disc is None:
                        print ('Loss at iteration {}/{}: {}'.format(iter, epoch_len, segmentation_loss))
                    else:
                        writer.add_scalar('Train/GeneratorLoss', g_loss, iter)
                        print("D_loss {}, G_loss {}, Seg loss at iteration {}/{}".format(d_loss, g_loss, segmentation_loss, iter, epoch_len))

                if iter % eval_every == 0:
                    mIOU = self.evaluate_meanIOU(self._val_loader, eval_debug)
                    if self.best_mIOU < mIOU:
                        self.best_mIOU = mIOU
                    self.save_model(epoch, self.best_mIOU, self.best_mIOU == mIOU)
                    writer.add_scalar('Train/MeanIOU', mIOU, iter)
                    print("Mean IOU at iteration {}/{}: {}".format(iter, epoch_len, mIOU))

                iter += 1


    def save_model(self, epoch, mIOU, is_best):
        save_dict = {
            'epoch': epoch + 1,
            'gen_dict': self._gen.state_dict(),
            'best_mIOU': mIOU,
            'gen_opt' : self._genoptimizer.state_dict()
        }
        if self._disc is not None:
            save_dict['disc_dict'] = self._disc.state_dict()
            save_dict['disc_opt'] = self._discoptimizer.state_dict()
            save_dict['gan_reg'] = self.gan_reg

        torch.save(save_dict, self.save_path)
        print ("=> Saved checkpoint '{}'".format(self.save_path))
        if is_best:
            shutil.copyfile(self.save_path, self.best_path)
            print ("=> Saved best checkpoint '{}'".format(self.best_path))

    def load_model(self):
        if os.path.isfile(self.save_path):
            print("=> loading checkpoint '{}'".format(self.save_path))
            checkpoint = torch.load(self.save_path)
            self.start_epoch = checkpoint['epoch']
            self.best_mIOU = checkpoint['best_mIOU']
            self._gen.load_state_dict(checkpoint['gen_dict'])
            self._genoptimizer.load_state_dict(checkpoint['gen_opt'])
            if self._disc is not None:
                self._disc.load_state_dict(checkpoint['disc_dict'])
                self._discoptimizer.load_state_dict(checkpoint['disc_opt'])
                self.gan_reg = checkpoint['gan_reg']

            print("=> loaded checkpoint '{}' (epoch {})".format(self.save_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.save_path))

    '''
    Evaluation methods
    '''
    def evaluate_pixel_accuracy(self, loader):
        true_pos = 0
        total_pix = 0
        for mini_batch_data, mini_batch_labels, _ in loader:
            mini_batch_prediction = self._gen(mini_batch_data)
            mini_batch_prediction = convert_to_mask(mini_batch_prediction)
            ## This assumes mini_batch_pred and mini_batch labels are of size B x C x H x W
            true_pos += torch.sum(mini_batch_prediction * mini_batch_labels).item()
            total_pix += torch.sum(mini_batch_labels).item()
        return float(true_pos) / total_pix

    def evaluate_pixel_mean_acc(self, loader):
        pix_acc = self.evaluate_pixel_accuracy(loader)
        return 1.0 / loader.dataset.numClasses * pix_acc

    def evaluate_meanIOU(self, loader, debug=False):
        print ("Evaluating mean IOU")
        self._gen.eval()
        if self._disc is not None:
            self._disc.eval()
        numClasses = loader.dataset.numClasses
        total = 0
        mIOU = 0.0
        iter = 0
        for data, mask_gt, _ in loader:
            data = data.to(self.device)
            batch_size = data.size()[0]
            total += batch_size
            mask_pred = convert_to_mask(self._gen(data))
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

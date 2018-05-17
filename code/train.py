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
            gan_reg=1.0, d_iters=5, weight_clip=1e-2, disc_lr=1e-5, gen_lr=1e-2,\
            experiment_dir='./', resume=False):
        """
        Training class for a specified model
        Args:
            net: (model) model to train
            train_loader: (DataLoader) train data
            val_load: (DataLoader) validation data
            gan_reg: Hyperparameter for the GAN loss (\lambda in the paper)
            experiment_dir: path to directory that saves everything
            resume: load from last saved checkpoint ?
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print ("Using device %s" % self.device)
        self._gen = generator.to(self.device)

        if discriminator is not None:
            self._disc = discriminator.to(self.device)
            self._discoptimizer = optim.Adam(self._disc.parameters(), lr=disc_lr) # Discriminator optimizer (needs to be separate)
        else:
            print ("Runing network without GAN loss.")
            self._disc = None
            self._discoptimizer = None
            self._BCEcriterion = None

        self._train_loader = train_loader
        self._val_loader = val_loader

        self._MCEcriterion = nn.CrossEntropyLoss() # Criterion for segmentation loss
        self._genoptimizer = optim.Adam(self._gen.parameters(), lr=gen_lr) # Generator optimizer

        self.gan_reg = gan_reg
        self.d_iters = d_iters
        self.start_iter = 0
        self.best_mIOU = 0
        self.weight_clip = weight_clip
        self.experiment_dir = experiment_dir
        self.save_path = os.path.join(experiment_dir, 'ckpt.pth.tar')
        self.best_path = os.path.join(experiment_dir, 'best.pth.tar')
        if resume:
            self.load_model()

    def _train_batch(self, mini_batch_data, mini_batch_labels, mini_batch_labels_flat, mode='disc'):
        """
        Performs one gradient step on a minibatch of data
        Args:
            mini_batch_data: (torch.Tensor) shape (N, C_in, H, W)
                where self._gen operates on (C_in, H, W) dimensional images
            mini_batch_labels: (torch.Tensor) shape (N, C_out, H, W)
                a batch of (H, W) binary masks for each of C_out classes
            mini_batch_labels_flat: (torch.Tensor) shape (N, H, W)
                a batch of (H, W) binary masks for each of C_out classes
            mode: discriminator or generator training
        Return:
            d_loss: (float) discriminator loss
            g_loss: (float) generator loss
            segmentation_loss: (float) segmentation loss
        """
        mini_batch_data = mini_batch_data.to(self.device) # Input image (B, 3, H, W)
        mini_batch_labels = mini_batch_labels.to(self.device).type(dtype=torch.float32) # Ground truth mask (B, C, H, W)
        mini_batch_labels_flat = mini_batch_labels_flat.to(self.device) # Groun truth mask flattened (B, H, W)
        gen_out = self._gen(mini_batch_data) # Segmentation output from generator (B, C, H , W)
        converted_mask = convert_to_mask(gen_out).to(self.device)

        if mode == 'disc' and self._disc is not None:
            d_loss = 0
            self._discoptimizer.zero_grad()
            scores_false = self._disc(mini_batch_data, converted_mask) # (B,)
            scores_true = self._disc(mini_batch_data, mini_batch_labels) # (B,)
            d_loss = torch.mean(scores_false) - torch.mean(scores_true)
            d_loss.backward()
            self._discoptimizer.step()
            # W-GAN weight clipping
            for p in self._disc.parameters():
                p.data.clamp_(-self.weight_clip, self.weight_clip)
            return d_loss, None
        if mode == 'gen':
            g_loss = 0
            self._genoptimizer.zero_grad()
            # GAN part
            if self._disc is not None:
                scores_false = self._disc(mini_batch_data, converted_mask)
                g_loss = -torch.mean(scores_false)
            # Minimize segmentation loss
            segmentation_loss = self._MCEcriterion(gen_out, mini_batch_labels_flat)
            gen_loss = segmentation_loss + self.gan_reg * g_loss
            gen_loss.backward()
            self._genoptimizer.step()
            return g_loss, segmentation_loss

    def train(self, num_epochs, print_every=100, eval_every=200, eval_debug=False):
        """
        Trains the model for a specified number of epochs
        Args:
            num_epochs: (int) number of epochs to train
            print_every: (int) number of minibatches to process before
                printing loss. default=100
        """
        writer = SummaryWriter(self.experiment_dir)

        iter = self.start_iter
        batch_size = self._train_loader.batch_size
        num_samples = len(self._train_loader.dataset)
        epoch_len = int(num_samples / batch_size)
        d_loss=0
        g_loss=0
        segmentation_loss=0
        for epoch in range(num_epochs):
            print ("Starting epoch {}".format(epoch))
            for mini_batch_data, mini_batch_labels, mini_batch_labels_flat in self._train_loader:
                if self._disc is not None:
                    for disc_train_iter in range(self.d_iters):
                        self._disc.train()
                        d_loss, _ = self._train_batch(
                                mini_batch_data, mini_batch_labels, mini_batch_labels_flat, 'disc')
                
                self._gen.train()
                g_loss, segmentation_loss = self._train_batch(
                        mini_batch_data, mini_batch_labels, mini_batch_labels_flat, 'gen')

                if iter % print_every == 0:
                    writer.add_scalar('Train/SegmentationLoss', segmentation_loss, iter)
                    if self._disc is None:
                        print ('Loss at iteration {}/{}: {}'.format(iter, epoch_len - 1, segmentation_loss))
                    else:
                        writer.add_scalar('Train/GeneratorLoss', g_loss, iter)
                        writer.add_scalar('Train/DiscriminatorLoss', d_loss, iter)
                        writer.add_scalar('Train/OverallLoss', self.gan_reg * d_loss + g_loss + segmentation_loss, iter)
                        print("D_loss {}, G_loss {}, Seg loss {} at iteration {}/{}".format(d_loss, g_loss, segmentation_loss, iter, epoch_len - 1))
                        print("Overall loss at iteration {} / {}: {}".format(iter, epoch_len - 1, self.gan_reg * d_loss + g_loss + segmentation_loss))
                if eval_every > 0 and iter % eval_every == 0:
                    # val_acc = self.evaluate_pixel_accuracy(self._val_loader)
                    # print ("Mean Pixel accuracy at iteration {}/{}: {}".format(iter, epoch_len, val_acc))
                    val_mIOU = self.evaluate_meanIOU(self._val_loader, eval_debug)
                    if self.best_mIOU < val_mIOU:
                        self.best_mIOU = val_mIOU
                    self.save_model(iter, self.best_mIOU, self.best_mIOU == val_mIOU)
                    writer.add_scalar('Val/MeanIOU', val_mIOU, iter)
                    print("Mean IOU at iteration {}/{}: {}".format(iter, epoch_len - 1, val_mIOU))
                iter += 1


    def save_model(self, iter, mIOU, is_best):
        save_dict = {
            'epoch': iter + 1,
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
            self.start_iter = checkpoint['iter']
            self.best_mIOU = checkpoint['best_mIOU']
            self._gen.load_state_dict(checkpoint['gen_dict'])
            self._genoptimizer.load_state_dict(checkpoint['gen_opt'])
            if self._disc is not None:
                self._disc.load_state_dict(checkpoint['disc_dict'])
                self._discoptimizer.load_state_dict(checkpoint['disc_opt'])
                self.gan_reg = checkpoint['gan_reg']

            print("=> loaded checkpoint '{}' (iter {})".format(self.save_path, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(self.save_path))

    '''
    Evaluation methods
    '''
    def evaluate_pixel_accuracy(self, loader):
        true_pos = 0
        total_pix = 0
        for mini_batch_data, mini_batch_labels, _ in loader:
            mini_batch_data = mini_batch_data.to(self.device)
            mini_batch_labels = mini_batch_labels.to(self.device).type(dtype=torch.float32)
            mini_batch_prediction = convert_to_mask(self._gen(mini_batch_data)).to(self.device)
            ## This assumes mini_batch_pred and mini_batch labels are of size B x C x H x W
            true_pos += torch.sum(mini_batch_prediction * mini_batch_labels).item()
            total_pix += torch.sum(mini_batch_labels).item()
        return float(true_pos) / (total_pix + 1e-12)

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
            mIOU += 1.0 / numClasses * torch.sum(classPresent * (truepositive / (totalpix + falsepos + 1e-12))).item()
            iter += 1
            if debug:
                print ("Processed %d batches out of %d, accumulated mIOU : %f" % (iter, len(loader), mIOU))
        return 1.0 / total * mIOU

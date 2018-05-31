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
            gan_reg=1.0, d_iters=5, weight_clip=1e-2, disc_lr=1e-5, gen_lr=1e-2, beta1=0.5,\
            train_gan=False, experiment_dir='./', resume=False):
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
        self.train_gan = train_gan
        if discriminator is not None and self.train_gan:
            print ("Training GAN")
            self._disc = discriminator.to(self.device)
            self._discoptimizer = optim.Adam(self._disc.parameters(), lr=disc_lr, betas=(beta1, 0.999)) # Discriminator optimizer (needs to be separate)
            self._BCEcriterion = nn.BCEWithLogitsLoss()
        else:
            print ("Runing network without GAN loss.")
            self._disc = None
            self._discoptimizer = None
            self._BCEcriterion = None

        self._train_loader = train_loader
        self._val_loader = val_loader

        self._MCEcriterion = nn.CrossEntropyLoss(self._train_loader.dataset.weights.to(self.device)) # Criterion for segmentation loss
        self._genoptimizer = optim.Adam(self._gen.parameters(), lr=gen_lr, betas=(beta1, 0.999)) # Generator optimizer
        self.gan_reg = gan_reg
        self.d_iters = d_iters
        self.start_iter = 0
        self.start_total_iters = 0
        self.start_epoch = 0
        self.best_mIOU = 0
        self.weight_clip = weight_clip
        self.experiment_dir = experiment_dir
        self.save_path = os.path.join(experiment_dir, 'ckpt.pth.tar')
        self.best_path = os.path.join(experiment_dir, 'best.pth.tar')
        if resume:
            self.load_model()

    def _train_batch(self, mini_batch_data, mini_batch_labels, mini_batch_labels_flat, mode):
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
        # mini_batch_data = mini_batch_data.to(self.device) # Input image (B, 3, H, W)
        mini_batch_data = mini_batch_data.to(self.device) # Input image (B, 3, H, W)

        mini_batch_labels = mini_batch_labels.to(self.device).type(dtype=torch.float32) # Ground truth mask (B, C, H, W)
        mini_batch_labels_flat = mini_batch_labels_flat.to(self.device) # Groun truth mask flattened (B, H, W)
        gen_out = self._gen(mini_batch_data) # Segmentation output from generator (B, C, H , W)
        converted_mask = convert_to_mask(gen_out).to(self.device)
        # false_labels = torch.zeros((mini_batch_data.size()[0], 1)).to(self.device)
        true_labels = torch.ones((mini_batch_data.size()[0], 1)).to(self.device)
        smooth_false_labels, smooth_true_labels = smooth_labels(mini_batch_data.size()[0], self.device)
        if mode == 'disc' and self._disc is not None and self.train_gan:
            d_loss = 0
            self._discoptimizer.zero_grad()
            scores_false = self._disc(mini_batch_data, converted_mask) # (B,)
            scores_true = self._disc(mini_batch_data, mini_batch_labels) # (B,)
            # d_loss = torch.mean(scores_false) - torch.mean(scores_true)
            d_loss = self._BCEcriterion(scores_false, smooth_false_labels) + self._BCEcriterion(scores_true, smooth_true_labels)
            d_loss.backward()
            self._discoptimizer.step()
            # W-GAN weight clipping
            # for p in self._disc.parameters():
            #     p.data.clamp_(-self.weight_clip, self.weight_clip)
            return d_loss, None
        if mode == 'gen':
            self._genoptimizer.zero_grad()
            g_loss = 0
            # GAN part
            if self._disc is not None and self.train_gan:
                scores_false = self._disc(mini_batch_data, converted_mask)
                g_loss = self._BCEcriterion(scores_false, true_labels)
                # g_loss = -torch.mean(scores_false)
            # Minimize segmentation loss
            segmentation_loss = self._MCEcriterion(gen_out, mini_batch_labels_flat)
            gen_loss = segmentation_loss + self.gan_reg * g_loss
            gen_loss.backward()
            self._genoptimizer.step()
            return g_loss, segmentation_loss

    def train(self, num_epochs, print_every=100, eval_every=500, eval_debug=False):
        """
        Trains the model for a specified number of epochs
        Args:
            num_epochs: (int) number of epochs to train
            print_every: (int) number of minibatches to process before
                printing loss. default=100
        """
        writer = SummaryWriter(self.experiment_dir)

        total_iters = self.start_total_iters
        iter = self.start_iter
        batch_size = self._train_loader.batch_size
        num_samples = len(self._train_loader.dataset)
        epoch_len = int(num_samples / batch_size)
        d_loss=0
        g_loss=0
        segmentation_loss=0
        d_iter = 0
        if total_iters is None:
            total_iters = iter + epoch_len * self.start_epoch
            print ("Total_iters starts at {}".format(total_iters))
        for epoch in range(self.start_epoch, num_epochs):
            print ("Starting epoch {}".format(epoch))
            for mini_batch_data, mini_batch_labels, mini_batch_labels_flat in self._train_loader:
                if self._disc is not None and self.train_gan and d_iter < self.d_iters:
                    self._disc.train()
                    d_loss, _ = self._train_batch(
                            mini_batch_data, mini_batch_labels, mini_batch_labels_flat, 'disc')
                    d_iter+= 1
                else:
                    self._gen.train()
                    g_loss, segmentation_loss = self._train_batch(
                            mini_batch_data, mini_batch_labels, mini_batch_labels_flat, 'gen')
                    d_iter=0
                writer.add_scalar('Train/SegmentationLoss', segmentation_loss, total_iters)
                gen_avg_grad_norm = average_grad_norm(self._gen)
                writer.add_scalar('Train/GeneratorAvgGradNorm', gen_avg_grad_norm, total_iters)
                if self._disc is not None and self.train_gan:
                    writer.add_scalar('Train/GeneratorLoss', g_loss, total_iters)
                    writer.add_scalar('Train/DiscriminatorLoss', d_loss, total_iters)
                    writer.add_scalar('Train/GanLoss', d_loss + g_loss, total_iters)
                    writer.add_scalar('Train/TotalLoss', self.gan_reg * (d_loss + g_loss) + segmentation_loss, total_iters)
                    disc_avg_grad_norm = average_grad_norm(self._disc)
                    writer.add_scalar('Train/DiscriminatorAvgGradNorm', disc_avg_grad_norm, total_iters)
                if total_iters % print_every == 0:
                    if self._disc is None or not self.train_gan:
                        print ('Loss at iteration {}/{}: {}'.format(iter, epoch_len - 1, segmentation_loss))
                    else:
                        print("D_loss {}, G_loss {}, Seg loss {} at iteration {}/{}".format(d_loss, g_loss, segmentation_loss, iter, epoch_len - 1))
                        print("Overall loss at iteration {} / {}: {}".format(iter, epoch_len - 1, self.gan_reg * (d_loss + g_loss) + segmentation_loss))
                if eval_every > 0 and total_iters % eval_every == 0:
                    val_mIOU = self.evaluate_meanIOU(self._val_loader, eval_debug)
                    if self.best_mIOU < val_mIOU:
                        self.best_mIOU = val_mIOU
                    self.save_model(iter, total_iters, epoch, self.best_mIOU, self.best_mIOU == val_mIOU)
                    writer.add_scalar('Val/MeanIOU', val_mIOU, total_iters)
                    print("Validation Mean IOU at iteration {}/{}: {}".format(iter, epoch_len - 1, val_mIOU))
                iter += 1
                total_iters += 1
            iter = 0


    def save_model(self, iter, total_iters, epoch, mIOU, is_best):
        save_dict = {
            'epoch': epoch,
            'iter': iter + 1,
            'total_iters': total_iters + 1,
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
            self.start_total_iters = checkpoint.get('total_iters', None)
            self.start_epoch = checkpoint['epoch']
            self.best_mIOU = checkpoint['best_mIOU']
            self._gen.load_state_dict(checkpoint['gen_dict'])
            self._genoptimizer.load_state_dict(checkpoint['gen_opt'])
            if self._disc is not None:
                if 'disc_dict' in checkpoint:
                  self._disc.load_state_dict(checkpoint['disc_dict'])
                  self._discoptimizer.load_state_dict(checkpoint['disc_opt'])
                  self.gan_reg = checkpoint['gan_reg']

            print("=> loaded checkpoint '{}' (iter {})".format(self.save_path, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(self.save_path))

    '''
    Evaluation methods
    '''
    def evaluate_pixel_accuracy(self, loader, ignore_background=False):
        true_pos = 0
        total_pix = 0
        for mini_batch_data, mini_batch_labels, _ in loader:
            mini_batch_data = mini_batch_data.to(self.device)
            mini_batch_labels = mini_batch_labels.to(self.device).type(dtype=torch.float32)
            mini_batch_prediction = convert_to_mask(self._gen(mini_batch_data)).to(self.device)
            ## This assumes mini_batch_pred and mini_batch labels are of size B x C x H x W
            print (mini_batch_prediction.size(), mini_batch_labels.size())

            if ignore_background:
                mini_batch_prediction, mini_batch_labels = mini_batch_prediction[:,:-1,:,:], mini_batch_labels[:,:-1,:,:]
            true_pos += torch.sum(mini_batch_prediction * mini_batch_labels).item()
            print (true_pos)
            total_pix += torch.sum(mini_batch_labels).item()
        print (true_pos)
        return float(true_pos) / (total_pix + 1e-12)

    def evaluate_pixel_mean_acc(self, loader):
        pix_acc = self.evaluate_pixel_accuracy(loader)
        return 1.0 / loader.dataset.numClasses * pix_acc

    def evaluate_meanIOU(self, loader, debug=False, ignore_background=True):
        print ("Evaluating mean IOU")
        self._gen.eval()
        if self._disc is not None:
            self._disc.eval()
        numClasses = loader.dataset.numClasses
        total = 0
        mIOU = 0.0
        iter = 0
        max_batches = 10
        for data, mask_gt, gt_visual in loader:
            data = data.to(self.device)
            batch_size = data.size()[0]
            total += batch_size
            mask_pred = convert_to_mask(self._gen(data))
            mask_gt = mask_gt.view((batch_size, numClasses, -1)).type(dtype=torch.float32).to(self.device)
            mask_pred = mask_pred.view((batch_size, numClasses, -1)).to(self.device)

            if ignore_background:
                mask_gt = mask_gt.narrow(1, 0, numClasses-1)
                mask_pred = mask_pred.narrow(1, 0, numClasses-1)
                #mask_gt, mask_pred = mask_gt[:,:-1,:], mask_pred[:,:-1,:]

            totalpix = torch.sum(mask_gt, 2)
            classPresent = (totalpix > 0).type(dtype=torch.float32) # Ignore class that was not originally present in the groundtruth
            truepositive = torch.sum(mask_gt * mask_pred, 2)
            falsepos = torch.sum(mask_pred, 2) - truepositive
            numerator = torch.sum(classPresent * (truepositive / (totalpix + falsepos + 1e-12)), 1)
            denominator = classPresent.sum(1)
            fraction = (numerator / denominator).masked_select(denominator > 0)
            mIOU +=  torch.sum(fraction).item()

            iter += 1
            if iter == max_batches:
                break
            if debug:
                print ("Processed %d batches out of %d, accumulated mIOU : %f" % (iter, len(loader), mIOU))
        return 1.0 / total * mIOU

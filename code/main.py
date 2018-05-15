import torch
from torch.utils.data import DataLoader

from train import Trainer
from model import *
from dataset import CocoStuffDataSet

import os, argparse, datetime
NUM_CLASSES = 11
SAVE_DIR = "../checkpoints" # Assuming this is launched from code/ subfolder.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--print_every', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='load model from saved checkpoint ')
    parser.add_argument('--experiment_name', '-n', type=str, default=None,
                        help='name of experiment used for saving loading checkpoints')

    parser.add_argument('--gan_reg', default=0.1, type=float,
                        help='Regularization strength from gan')
    parser.add_argument('-d', '--d_iters', default=5, type=int,
                        help='Number of training iterations for discriminator within one loop')

    args = parser.parse_args()
    batch_size = args.batch_size

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    ### Create experiment specific directory
    EXPERIMENT_DIR = SAVE_DIR + '/'
    if args.experiment_name is not None:
        EXPERIMENT_DIR += args.experiment_name
    else:
        now = datetime.datetime.now()
        EXPERIMENT_DIR += now.strftime("%m_%d_%H%M")

    save_path = EXPERIMENT_DIR + '/ckpt.pth.tar'
    best_path = EXPERIMENT_DIR + '/best.pth.tar'

    # net = SegNetSmall(num_classes, pretrained=True)
    generator = VerySmallNet(NUM_CLASSES)
    discriminator = None
    # net = SegNetSmaller(NUM_CLASSES, pretrained=True)
    train_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='train'), args.batch_size, shuffle=True)
    val_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='val'), args.batch_size, shuffle=False)

    trainer = Trainer(generator, discriminator, train_loader, val_loader, \
            gan_reg=args.gan_reg, d_iters=args.d_iters, save_path=save_path, best_path=best_path, resume=args.load_model)

    trainer.train(num_epochs=args.epochs, print_every=args.print_every)

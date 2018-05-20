import torch
from torch.utils.data import DataLoader
from train import Trainer
from generator import *
from discriminator import GAN
from dataset import CocoStuffDataSet
import os, argparse, datetime, json

NUM_CLASSES = 11
SAVE_DIR = "../checkpoints" # Assuming this is launched from code/ subfolder.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--mode', default='train', type=str,
                        help='Mode train/eval')
    # Training parameters
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('-s', '--size', default=128, type=int,
                        help='size of images (default:128)')
    # Utility parameters
    parser.add_argument('--print_every', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval_every', '-e', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='load model from saved checkpoint ')
    parser.add_argument('--experiment_name', '-n', type=str, default=None,
                        help='name of experiment used for saving loading checkpoints')
    # GAN Hyperparameters
    parser.add_argument('--disc_lr', default=1e-2, type=float,
                        help='Learning rate for discriminator')
    parser.add_argument('--gen_lr', default=0.1, type=float,
                        help='Learning rate for generator')
    parser.add_argument('--weight_clip', default=0.01, type=float,
                        help='Weight clipping for W-GAN loss')
    parser.add_argument('--gan_reg', default=0.1, type=float,
                        help='Regularization strength from gan')
    parser.add_argument('-d', '--d_iters', default=5, type=int,
                        help='Number of training iterations for discriminator within one loop')

    args = parser.parse_args()
    batch_size = args.batch_size

    # Create experiment specific directory
    if args.experiment_name is not None:
        experiment_dir = os.path.join(SAVE_DIR, args.experiment_name)
    else:
        now = datetime.datetime.now()
        experiment_dir = os.path.join(SAVE_DIR, now.strftime("%m_%d_%H%M"))

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    if not args.load_model:
        with open(experiment_dir+'/args.json', 'w') as outfile:
            json.dump(vars(args), outfile, sort_keys=True, indent=4)

    HEIGHT, WIDTH = args.size, args.size
    image_shape = (3, HEIGHT, WIDTH)
    segmentation_shape = (NUM_CLASSES, HEIGHT, WIDTH)
    # generator = VerySmallNet(NUM_CLASSES)
    discriminator = None
    # generator = SegNetSmaller(NUM_CLASSES, pretrained=True)
    generator = SegNet16(NUM_CLASSES, pretrained=True)
    discriminator = GAN(NUM_CLASSES, segmentation_shape, image_shape)

    val_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='val', height=HEIGHT, width=WIDTH),
                                args.batch_size, shuffle=False)
    train_loader = DataLoader(CocoStuffDataSet(supercategories=['animal'], mode='train', height=HEIGHT, width=WIDTH),
                                args.batch_size, shuffle=True)
    trainer = Trainer(generator, discriminator, train_loader, val_loader, \
                    gan_reg=args.gan_reg, d_iters=args.d_iters, \
                    weight_clip= args.weight_clip, disc_lr=args.disc_lr, gen_lr=args.gen_lr, \
                    experiment_dir=experiment_dir, resume=args.load_model)

    if args.mode == "train":
        trainer.train(num_epochs=args.epochs, print_every=args.print_every, eval_every=args.eval_every)
    elif args.mode == 'eval':
        assert(args.load_model), "Need to load model to evaluate it"
        # just do evaluation
        print ('mIOU {}'.format(trainer.evaluate_meanIOU(val_loader, debug=True)))

import torch
from torch.utils.data import DataLoader
from train import Trainer
from generator import get_generator
from discriminator import GAN
from dataset import CocoStuffDataSet
import os, argparse, datetime, json

SAVE_DIR = "../checkpoints" # Assuming this is launched from code/ subfolder.

def by_pixel_weights(dataloader, savename):
    num_classes = dataloader.dataset.numClasses
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    counts = torch.zeros(num_classes).float().to(device)
    for _, masks, _ in dataloader:
        masks = masks.float().to(device)
        counts += (masks.view((num_classes, -1))).sum(dim=1)
    weights = counts.reciprocal()
    weights /= weights.sum()

    print("Saving weights to ", savename)
    torch.save(weights, savename)
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--mode', default='train', type=str,
                        help='Mode train/eval')
    # Training parameters
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('-s', '--size', default=128, type=int,
                        help='size of images (default:128)')
    # Utility parameters
    parser.add_argument('--print_every', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--eval_every', '-e', default=300, type=int,
                        metavar='N', help='eval frequency (default: 300)')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='load model from checkpoint ')
    parser.add_argument('--load_iter', '-li', type=int, default=None,
                        help='specify which iter to resume training from')
    parser.add_argument('--experiment_name', '-n', type=str, default=None,
                        help='name of experiment used for saving loading checkpoints')
    # GAN Hyperparameters
    parser.add_argument('--train_gan', type=bool, default=False,
                        help='decide whether to train GAN')
    parser.add_argument('--disc_lr', default=1e-5, type=float,
                        help='Learning rate for discriminator')
    parser.add_argument('--gen_lr', default=1e-4, type=float,
                        help='Learning rate for generator')
    parser.add_argument('--weight_clip', default=0.01, type=float,
                        help='Weight clipping for W-GAN loss')
    parser.add_argument('--gan_reg', default=1.0, type=float,
                        help='Regularization strength from gan')
    parser.add_argument('-d', '--d_iters', default=5, type=int,
                        help='Number of training iterations for discriminator within one loop')
    parser.add_argument('-g', '--g_iters', default=5, type=int,
                        help='Number of training iterations for generator within one loop')
    parser.add_argument('--generator_name', default='SegNet16', type=str,
                        help='Name of generator model to run')
    parser.add_argument('--use_bn', default='True', type=bool,
                        help='Use batch norm in Decoder block')

    args = parser.parse_args()

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
    else:
        with open(experiment_dir+'/args.json', 'r') as infile:
            args_dict = json.load(infile)
            args_dict['load_model'] = True
            args_dict['experiment_name'] = args.experiment_name
            args_dict['train_gan'] = args.train_gan
            current_dict = vars(args)
            for (key, value) in args_dict.items():
                current_dict[key] = value
            args = argparse.Namespace(**current_dict)

    HEIGHT = WIDTH = args.size
    val_dataset = CocoStuffDataSet(mode='val', supercategories=['animal'], height=HEIGHT, width=WIDTH, do_normalize=False)
    train_dataset = CocoStuffDataSet(mode='train', supercategories=['animal'], height=HEIGHT, width=WIDTH, do_normalize=False)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    NUM_CLASSES = train_dataset.numClasses
    print ("Number of classes: {}".format(NUM_CLASSES))
    image_shape = (3, HEIGHT, WIDTH)
    segmentation_shape = (NUM_CLASSES, HEIGHT, WIDTH)

    discriminator = None
    generator = get_generator(args.generator_name, NUM_CLASSES, args.use_bn)
    if args.train_gan:
        discriminator = GAN(NUM_CLASSES, segmentation_shape, image_shape)

    trainer = Trainer(generator, discriminator, train_loader, val_loader, \
                    gan_reg=args.gan_reg, d_iters=args.d_iters, g_iters=args.g_iters,\
                    weight_clip= args.weight_clip, disc_lr=args.disc_lr, gen_lr=args.gen_lr,
                    train_gan= args.train_gan, \
                    experiment_dir=experiment_dir, resume=args.load_model, load_iter=args.load_iter)

    if args.mode == "train":
        trainer.train(num_epochs=args.epochs, print_every=args.print_every, eval_every=args.eval_every)
    elif args.mode == 'eval':
        assert(args.load_model), "Need to load model to evaluate it"
        # just do evaluation
        print (trainer.get_confusion_matrix(val_loader))
        print ('mIOU {}'.format(trainer.evaluate_meanIOU(val_loader)))

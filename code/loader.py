import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

class CocoStuffLoader(dset.CocoDetection):
    def __init__(
            self, img_dir='../cocostuff/images/',
            annot_dir='../cocostuff/annotations/',
            mode='train'):
        super(CocoStuffLoader, self).__init__(
            root=img_dir + mode + '2017/',
            annFile=annot_dir+'instances_'+mode+'2017.json',
            transform=transforms.ToTensor())

        print('Loaded %d samples: ' % len(self))

    def display(self, img_id):
        img, target = self[img_id]
        print("Image Size: ", img.size())
        display_image = np.transpose(img.numpy(), (1, 2, 0))
        plt.figure()
        plt.subplot(121)
        plt.imshow(display_image)
        plt.axis('off')
        plt.title('original image')
        plt.subplot(122)
        plt.imshow(display_image)
        self.coco.showAnns(target)
        plt.axis('off')
        plt.title('annotated image')
        plt.show()

cocostuff = CocoStuffLoader()
cocostuff.display(0)

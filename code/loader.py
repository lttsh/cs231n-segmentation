import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

class CocoStuffDataSet(dset.CocoDetection):
    '''
    Custom dataset
    '''
    def __init__(
            self, img_dir='../cocostuff/images/',
            annot_dir='../cocostuff/annotations/',
            mode='train', categories=None, supercategories=None):
        super().__init__(
            root=img_dir + mode + '2017/',
            annFile=annot_dir+'instances_'+mode+'2017.json',
            transform=transforms.ToTensor())
        self.cats = categories ## List of categories to load
        if self.cats is not None:
            catIds = self.coco.getCatIds(catNms=self.cats)
            self.ids=[]
            for id in catIds:
                self.ids += self.coco.getImgIds(catIds=[id])
                self.ids = list(set(self.ids))

        self.supercats = supercategories
        if self.supercats is not None:
            catIds = self.coco.getCatIds(supNms=self.supercats)
            self.ids=[]
            for id in catIds:
                self.ids += self.coco.getImgIds(catIds=[id])
                self.ids = list(set(self.ids))
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

cocostuff = CocoStuffDataSet(supercategories=['animal'])
cocostuff.display(0)

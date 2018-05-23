import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from PIL import Image
import os
from utils import discrete_cmap

class CocoStuffDataSet(dset.CocoDetection):
    '''
    Custom dataset handler for MSCOCO Detection dataset
    categories/supercategories: list of categories needed.
    '''
    def __init__(
            self, img_dir='../cocostuff/images/',
            annot_dir='../cocostuff/annotations/',
            mode='train', height=256, width=256,
            categories=None, supercategories=None,
            ):
        if width is None or height is None:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor()
            ])
        super().__init__(
            root=img_dir + mode + '2017/',
            annFile=annot_dir+'instances_'+mode+'2017.json',
            transform=transform)

        self.width = width # Resize width
        self.height = height # Resize height
        self.cats = categories # Categories to load
        self.supercats = supercategories # Super categories to load

        if self.cats is not None:
            self.catIds = self.coco.getCatIds(catNms=self.cats)
            self.ids=[]
            for id in self.catIds:
                self.ids += self.coco.getImgIds(catIds=[id])
                self.ids = list(set(self.ids))

        if self.supercats is not None:
            self.catIds = self.coco.getCatIds(supNms=self.supercats) # Categories ID to be used
            self.ids=[] # Images ID containing the categories
            for id in self.catIds:
                self.ids += self.coco.getImgIds(catIds=[id])
                self.ids = list(set(self.ids))
        else:
            self.catIds = self.coco.getCatIds()
        self.numClasses = len(self.catIds) + 1
        print('Loaded %d samples: ' % len(self))

        weights = np.zeros(self.numClasses)
        for idx, catId in enumerate(self.catIds):
            ids = self.coco.getImgIds(catIds=catId)
            weight = 1.0 / float(len(ids) + 1e-8)
            weights[idx] = weight
        weights[-1] = 1.0 / float(len(self) + 1e-8)
        weights /= np.sum(weights)
        self.weights = torch.Tensor(weights)
        print (self.weights)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, mask).
                'image' ND array of size (3, H, W)
                'mask' ND array of size (C + 1, H, W) where C is the number of categories ( + 1 for background category)
                'mask_flat' ND array of size (H, W) where each pixel has value between 0 and C depending on their class
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        masks = np.zeros((self.numClasses, self.width, self.height))
        for ann in target:
            if ann['category_id'] in self.catIds:
                masks[self.catIds.index(ann['category_id'])] += \
                    misc.imresize(self.coco.annToMask(ann), (self.width, self.height))

        # Create background mask
        masks[-1] = 1 - np.sum(masks[:-1, : , :], 0)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return img, masks, np.argmax(masks, axis=0)

    def gather_stats(self):
        images = self.coco.dataset['images']
        heights = np.zeros((len(images),))
        widths = np.zeros((len(images),))
        for i, image in enumerate(images):
            heights[i] = image['height']
            widths[i] = image['width']
        self.min_height = np.min(heights)
        self.max_height = np.max(heights)
        self.min_width = np.min(widths)
        self.max_width = np.max(widths)
        print('Min height %d, max height %d' % (self.min_height, self.max_height))
        print('Min width %d, max width %d' % (self.min_width, self.max_width))
        print ("There are %f of images at max height and %f at max width" %
            (np.mean(heights==self.max_height), np.mean(widths==self.max_width)))

        for catId in self.catIds:
            ids = self.coco.getImgIds(catIds=catId)
            print ("%f images contain category %d" % (float(len(ids))/float(len(self)), catId))

    def display(self, img_id):
        img, _, masks = self[img_id]
        print("Image Size: ", img.size())
        display_image = np.transpose(img.numpy(), (1, 2, 0))
        plt.figure()
        plt.subplot(121)
        plt.imshow(display_image)
        plt.axis('off')
        plt.title('original image')
        plt.subplot(122)
        plt.imshow(display_image)
        print (np.unique(masks))
        cmap = discrete_cmap(self.numClasses, 'Paired')
        norm = colors.NoNorm(vmin=0, vmax=self.numClasses)
        plt.imshow(masks, alpha=0.8, cmap=cmap, norm=norm)
        plt.axis('off')
        plt.title('annotated image')
        plt.show()

if __name__ == "__main__":
    ## Display
    cocostuff = CocoStuffDataSet(supercategories=['animal'])
    for _ in range(10):
        cocostuff.display(np.random.randint(low=0, high=len(cocostuff)))
    cocostuff.gather_stats()

import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class Flickr8k(data.Dataset):
    def __init__(self, root='../data/flickr8k', train=True, transform=None):
        super().__init__()

        self.path = { 'imgs': os.path.join(root, 'Flicker8k_Dataset'),
                      'captions': os.path.join(root, 'Flickr8k_text', 'Flickr8k.token.txt'),}
        if train:
            self.path['ids'] = os.path.join(root, 'Flickr8k_text', 'Flickr_8k.trainImages.txt')
        else:
            self.path['ids'] = os.path.join(root, 'Flickr8k_text', 'Flickr_8k.testImages.txt')

        self.ids = self.load_all_img_ids()
        # self.imgs = self.load_all_imgs()
        self.captions = self.load_all_captions()

        self.transform = transform
        self.img_size = 224
        self.img_mean = [122.7717, 115.9465, 102.9801]

    def __getitem__(self, index):
        id = self.ids[index]
        caption = self.captions[id]
        img = self.img_loader(os.path.join(self.path['imgs'], id))
        if self.transform is not None:
            img = self.transform(img)
        return id, img, caption

    def __len__(self):
        return len(self.ids)

    def img_loader(self, path):
        """
        Load image using cv2
        """
        return cv2.imread(path)

    def resize_img(self, img):
        """
        Resize image to fixed size
        """
        return cv2.resize(img, (self.img_size, self.img_size))

    def sub_mean(self, img):
        pass

    def load_all_img_ids(self):
        """
        Load all image ids from file
        """
        ids = []
        with open(self.path['ids']) as f:
            lines = f.readlines()
            for id in lines:
                ids.append(id[:-1])
        return ids
    
    def load_all_imgs(self):
        """
        Load all image according to ids
        """
        imgs = []
        for id in self.ids:
            imgs.append(os.path.join(self.path['imgs'], id))
        return imgs

    def load_all_captions(self):
        """
        Load all captions in file and save to dict: {id: [captions]}
        """
        captions = {}
        with open(self.path['captions']) as f:
            for line in f:
                split = line.split('#')
                id = split[0]
                caption = split[1][2:-3]
                
                if id not in captions:
                    captions[id] = [caption]
                else:
                    captions[id].append(caption)
        
        # assert len(captions) == len(self.ids), \
        #         '# captions != # ids: {} != {}'.\
        #         format(len(captions), len(self.ids))

        return captions

    def test_ids(self):
        return self.ids

    def test_imgs(self):
        import matplotlib.pyplot as plt
        img = self.img_loader(self.imgs[10])
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = Flickr8k(transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    for i, data in enumerate(dataloader, 0):
        id, img, caption = data
        img = img.numpy().squeeze().transpose(1, 2, 0)
        print('id: ', id)
        print('caption', caption)
        plt.imshow(img)
        plt.show()
        break

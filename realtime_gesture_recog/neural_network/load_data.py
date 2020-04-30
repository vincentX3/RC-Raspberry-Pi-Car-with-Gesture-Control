import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms
import os
from utils import PATH_DATA


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

class GestureData(Dataset):
    # image named as 'label_id.png'

    def __init__(self,root_dir=PATH_DATA, transform = None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        filenames = os.listdir(self.root_dir)
        filename = filenames[idx]
        img = io.imread(os.path.join(self.root_dir, filename))
        # get image's label
        label = np.array([int(filename[0])])
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

'''
how to use the class?
just call

data = GestureData(transform=transforms.Compose([Rescale(256),ToTensor()])

then pass to a dataloader.
'''

if __name__ == '__main__':
    # check
    g_test = GestureData(transform=transforms.Compose([Rescale(256),ToTensor()]))
    print(len(g_test))
    for i in range(3):
        sample = g_test[i]
        # print(sample)
        print(i, sample['image'].size() ,sample['label'].size())
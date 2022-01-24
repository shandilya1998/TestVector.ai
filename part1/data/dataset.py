import os
import torch
import numpy as np
import torchvision
from constants import params
import cv2
from utils import mnist_reader

torch.manual_seed(params['manual_seed'])

class FashionMNIST(torch.utils.data.Dataset):
    def __init__(self, params, kind = 'train'):
        path = 'data/train'
        self.batch_size = params['batch_size']
        self.shape = (params['image_height'], params['image_width'])
        self.to_shuffle = params['shuffle_dataset']
        self.to_rotate = params['rotate_images']
        self.to_normalise = params['normalise_images']
        self.is_color = params['is_image_color']
        if self.is_color:
            assert len(params['normalise_mean']) == 3 and len(params['normalise_std']) == 3
        transforms = []
        if self.to_rotate:
            transforms.append(
                torchvision.transforms.RandomRotation(
                    180,

                )
            )
        if self.to_normalise:
            transforms.append(
                torchvision.transforms.Normalize(
                    mean=params['normalise_mean'],
                    std=params['normalise_std']
                )
            )
        self.transforms = torch.nn.Sequential(
            *transforms
        )
        self.x, self.y = mnist_reader.load_mnist(path, kind = kind)

    def img_transform(self, img):
        # 0-255 to 0-1
        img = torch.from_numpy(np.float32(img.copy()) / 255.)
        img = self.transforms(img)
        return img

    def __len__(self):
        assert self.x.shape[0] == self.y.shape[0]
        return self.x.shape[0]

    def __getitem__(self, index):
        if not self.is_color:
            # Color images are stored as (H, W, C) arrays
            # Grayscale images are flattened
            # Both of which are transformed to (C, H, W)
            images = self.x[index].reshape(self.batch_size, 1, self.shape[0], self.shape[1])
        else:
            images = self.x[index].transpose(2, 0, 1)
        labels = torch.from_numpy(self.y[start: end].copy())
        images = self.img_transform(images)
        return images, labels

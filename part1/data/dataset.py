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
        path = 'data/{}'.format(kind)
        self.shape = (params['image_height'], params['image_width'])
        self.to_rotate = params['rotate_images']
        self.to_normalise = params['normalise_images']
        self.is_color = params['is_image_color']
        self.to_transform = self.to_rotate or self.to_normalise
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
        if params['additional_transforms']:
            transforms.extend(params['additional_transforms'])
        self.transforms = torch.nn.Sequential(
            *transforms
        )
        x, y = mnist_reader.load_mnist(path, kind = kind)
        self.data = list(zip(x, y))

    def img_transform(self, img):
        # 0-255 to 0-1
        if self.to_transform:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if not self.is_color:
            # Color images are stored as (H, W, C) arrays
            # Grayscale images are flattened
            # Both of which are transformed to (C, H, W)
            image = image.reshape(1, self.shape[0], self.shape[1])
        else:
            image = image.transpose(2, 0, 1)
        image = image.copy().astype(np.float32) / 255.0
        image = self.img_transform(torch.from_numpy(image))
        return image, torch.from_numpy(np.array([label], dtype = np.uint8))

def get_dataloader(params, kind):
    dataset = FashionMNIST(params, kind)
    return torch.utils.data.DataLoader(
        dataset,
        params['batch_size'],
        shuffle = params['shuffle_dataset']
    )

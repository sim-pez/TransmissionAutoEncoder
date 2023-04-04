import os
import torch
import re
import numpy as np
import torch.nn as nn
from torchvision.transforms.functional import normalize


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)


def find_images(path):
    '''
    Find all images in a directory and its subdirectories.
    '''
    images = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            ext = os.path.splitext(filename)[-1].lower()
            if ext == ".png":
                relative_path = os.path.relpath(os.path.join(dirpath, filename), path)
                images.append(relative_path)
    return images


def find_device_and_batch_size():
    '''
    Finds if a gpu is available and returns also batch size
    '''
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        batch_size = 4
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size = 8
    else:
        device = torch.device("cpu")
        batch_size = 4

    return device, batch_size


def get_encoding_size(checkpoint_name):
    '''
    Get model encoding size from the checkpoint name
    '''
    encoding_size = checkpoint_name.split('/')[-2].split('-')[-1]
    return int(encoding_size)


intmap = {0.0000 : 0,
          0.0039 : 1,
          0.0078 : 2,
          0.0118 : 3,
          0.0157 : 4,
          0.0196 : 5,
          0.0235 : 6,
          0.0275 : 7,
          0.0314 : 8,
          0.0353 : 9,
          0.0392 : 10,
          0.0431 : 11,
          0.0471 : 12,
          0.0510 : 13,
          0.0549 : 14,
          0.0588 : 15,
          0.0627 : 16,
          0.0667 : 17,
          0.0706 : 18,
          0.0745 : 19}


def map_values_tensor(input_tensor):
    '''
    Returns a tensor with the mapped values
    '''

    output = torch.zeros_like(input_tensor, dtype=torch.int64)
    for value, label_class in intmap.items():
        output = torch.where(input_tensor == value, torch.tensor(label_class), output)
    return output
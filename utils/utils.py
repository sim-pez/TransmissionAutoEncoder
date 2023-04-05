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


intmap = {0.0 : 0,
          0.003921568859368563 : 1,
          0.007843137718737125 : 2,
          0.0117647061124444 : 3,
          0.01568627543747425 : 4,
          0.019607843831181526 : 5,
          0.0235294122248888 : 6,
          0.027450980618596077 : 7,
          0.0313725508749485 : 8,
          0.03529411926865578 : 9,
          0.03921568766236305 : 10,
          0.04313725605607033 : 11,
          0.0470588244497776 : 12,
          0.05098039284348488 : 13,
          0.054901961237192154 : 14,
          0.05882352963089943 : 15,
          0.062745101749897 : 16,
          0.06666667014360428 : 17,
          0.07058823853731155 : 18,
          0.07450980693 : 19} # 19 is not in the dataset



def map_values_tensor(input_tensor):
    '''
    Returns a tensor with the mapped values
    '''

    output = torch.zeros_like(input_tensor, dtype=torch.int64)

    for value, label_class in intmap.items():
        output = torch.where(input_tensor == value, torch.tensor(label_class), output)
    return output
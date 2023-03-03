import os
import torch

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
        batch_size = 32
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

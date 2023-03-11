import os
import torch
import glob

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


value2class = { 0.003921568859368563  : 0, 
                0.007843137718737125  : 1, 
                0.0117647061124444    : 2, 
                0.01568627543747425   : 3, 
                0.019607843831181526  : 4, 
                0.0235294122248888    : 5, 
                0.027450980618596077  : 6, 
                0.0313725508749485    : 7, 
                0.03529411926865578   : 8, 
                0.03921568766236305   : 9, 
                0.04313725605607033  : 10,
                0.0470588244497776   : 11,
                0.05098039284348488  : 12,
                0.05490196123719215  : 13,
                0.05882352963089943  : 14,
                0.062745101749897    : 15,
                0.06666667014360428  : 16,
                0.07058823853731155  : 17,
                0.07450980693101883  : 18,
                0.0784313753247261   : 19,
                0.08235294371843338  : 20,
                0.08627451211214066  : 21,
                0.09019608050584793  : 22,
                0.0941176488995552   : 23,
                0.09803921729326248  : 24,
                0.10196078568696976  : 25,
                0.10588235408067703  : 26,
                0.10980392247438431  : 27,
                0.11372549086809158  : 28,
                0.11764705926179886  : 29,
                0.12156862765550613  : 30,
                0.125490203499794    : 31,
                0.12941177189350128  : 32,
                0.13333334028720856  : 33,
                0.13725490868091583  : 34}


def map_values_tensor(input_tensor):
    '''
    Returns a tensor with the mapped values
    '''

    output = torch.zeros_like(input_tensor, dtype=torch.int64)
    for value, label_class in value2class.items():
        output = torch.where(input_tensor == value, torch.tensor(label_class), output)
    return output

def get_checkpoint_dir(mode, encoding_size=None, l=None):
    '''
    Get checkpoint directory from the checkpoint name
    '''
    if mode == 'complete':
        checkpoint_dir = f"checkpoints/mode:[{mode}]  enc:[{encoding_size}]  l:[{l}]"
    elif mode == 'autoencoder_only':
        checkpoint_dir = f"checkpoints/mode:[{mode}]  enc:[{encoding_size}]"
    elif mode == 'segmentation_only':
        checkpoint_dir = f"checkpoints/mode:[{mode}]"
    else:
        raise ValueError("Invalid mode")
    return checkpoint_dir


def get_last_checkpoint(checkpoint_dir):
    '''
    Get last checkpoint from the checkpoint directory
    '''
    pattern = 'epoch:[*.pth'
    file_paths = glob.glob(os.path.join(checkpoint_dir, pattern))
    file_paths.sort(key=os.path.getmtime, reverse=True)
    last_checkpoint_path = file_paths[0]
    return last_checkpoint_path
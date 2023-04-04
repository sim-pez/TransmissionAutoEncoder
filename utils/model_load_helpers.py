import os
import re

def get_checkpoint_dir(mode, encoding_size=None, l=None):
    '''
    Get checkpoint directory from the checkpoint name
    '''
    if mode == 'complete':
        checkpoint_dir = f"checkpoints/mode:[{mode}]  enc:[{encoding_size}]  r:[{l}]"
    elif mode == 'image_only':
        checkpoint_dir = f"checkpoints/mode:[{mode}]  enc:[{encoding_size}]"
    else:
        raise ValueError("Invalid mode")
    return checkpoint_dir


def get_last_checkpoint(checkpoint_dir):
    '''
    Get last checkpoint from the checkpoint directory
    '''
    file_paths = os.listdir(checkpoint_dir)
    if not file_paths:
        return None
    file_paths.sort(reverse=True)
    last_checkpoint_name = file_paths[0]
    last_checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_name)

    return last_checkpoint_path


def get_parameters_from_checkpoint(checkpoint_path):
    '''
    Get parameters from the checkpoint name
    '''
    mode = re.findall(r'mode:\[(\w+)\]', checkpoint_path)[0]
    encoding_size = int(re.findall(r'enc:\[(\d+)\]', checkpoint_path)[0])

    return mode, encoding_size
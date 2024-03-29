import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from model import Autoencoder
from dataloader import ImageDataset
from dataset.cityscapes import Cityscapes
from utils.utils import find_device_and_batch_size
from utils.model_load_helpers import get_parameters_from_checkpoint


def get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='dataset',
                        help="Path to the dataset directory. If not specified, it will use the 'dataset' folder")
    parser.add_argument("--checkpoin_path", type=str, default='checkpoints/mode:[complete]  enc:[32]  r:[0.8]/epoch:[199]  test:[0.05272]  train:[0.05247].pth',
                        help='Model relative path')

    return parser


def example(dataset_path, checkpoint_path):
    '''
    Do an example usage of the model. The output will be saved in the output folder.
    If not checkpoint path is provided, the last checkpoint will be used.
    '''

    #load model
    mode, encoding_size = get_parameters_from_checkpoint(checkpoint_path)
    device, batch_size = find_device_and_batch_size()
    model = Autoencoder(mode=mode, encoding_size=encoding_size)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print('Model loaded from checkpoint ' + checkpoint_path)

    dataset = ImageDataset(dataset_path=dataset_path, mode='test', get_img_path=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    
    start_index = checkpoint_path.find('/') + 1
    end_index = checkpoint_path.find('/', start_index)
    filename = os.path.join('output', checkpoint_path[start_index:end_index])
    

    with torch.no_grad():
        for imgs, segm, paths in tqdm(dataloader):
            imgs = imgs.to(device)
            segm = segm.to(device)
            x = torch.cat((imgs, segm), dim=1)
            output_imgs, output_segs = model(x)

            basename_list = []
            full_dirpath_list = []
            for path in paths:
                basename_list.append(os.path.basename(path))
                full_dirpath_list.append(os.path.dirname(os.path.join(filename, path)))
                os.makedirs(full_dirpath_list[-1], exist_ok=True)
            
            for img, full_dirpath, basename in zip(output_imgs, full_dirpath_list, basename_list):
                save_image(img, f'{full_dirpath}/{basename}_img.png')

            if mode == 'complete':
                for seg, full_dirpath, basename in zip(output_segs, full_dirpath_list, basename_list):
                    seg = torch.argmax(seg, dim=0).cpu().numpy()
                    seg = Cityscapes.decode_target(seg)
                    seg = Image.fromarray(seg.astype('uint8'))
                    seg.save(f'{full_dirpath}/{basename}_seg.png')

    print(f"Done!")


if __name__ == "__main__":

    opts = get_argparser().parse_args()
    
    example(opts.dataset, opts.checkpoint_path)
import os
import torch
import warnings
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms as T
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import DataLoader


from utils.utils import Denormalize
from model import Autoencoder
from dataset.cityscapes import Cityscapes
from utils.model_load_helpers import get_parameters_from_checkpoint
from utils.utils import find_device_and_batch_size
from dataloader import ImageDataset



checkpoint_path = 'checkpoints/mode:[complete]  enc:[32]  r:[0.8]/epoch:[001]  test:[0.05548]  train:[0.05500].pth'

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


    dataset = ImageDataset(dataset_path=dataset_path, mode="test")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    
    start_index = checkpoint_path.find('/') + 1
    end_index = checkpoint_path.find('/', start_index)
    filename = checkpoint_path[start_index:end_index]
    os.makedirs(os.path.join('output' ,filename), exist_ok=True)

    with torch.no_grad():
        for i, (imgs, segm) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(device)#(device, dtype=torch.float32)
            segm = segm.to(device)#(device, dtype=torch.float32)
            x = torch.cat((imgs, segm), dim=1)
            output_imgs, output_segs = model(x)
            
            for j, img in enumerate(output_imgs):                
                save_image(img, f'output/{filename}/{str(i*batch_size+j)}_img.png')

            if mode == 'complete':
                for j, seg in enumerate(output_segs):
                    seg = torch.argmax(seg, dim=0).cpu().numpy()
                    seg = Cityscapes.decode_target(seg)
                    seg = Image.fromarray(seg.astype('uint8'))
                    seg.save(f'output/{filename}/{str(i*batch_size+j)}_seg.png')

    print(f"Done!")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    dataset_path = 'dataset'

    example(dataset_path, checkpoint_path=checkpoint_path)
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


img_path = 'dataset/leftImg8bit/test/lindau/lindau_000000_000019_rightImg8bit.png'
checkpoint_path = 'checkpoints/mode:[complete]  enc:[32]  r:[0.8]/epoch:[001]  test:[0.15604]  train:[0.15178].pth'

def example(img_path, checkpoint_path):
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

    dataset = Cityscapes(root='dataset', segmentation_folder='resnet_seg_nocolor', split='train')
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)  # drop_last=True to ignore single-image batches.
    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
    start_index = checkpoint_path.find('/') + 1
    end_index = checkpoint_path.find('/', start_index)
    filename = checkpoint_path[start_index:end_index]
    filename = 'zzz' #TODO delete
    os.makedirs(os.path.join('output' ,filename), exist_ok=True)

    img_count = 0
    with torch.no_grad():
        for imgs, segm in tqdm(dataloader):
            segm = F.one_hot(segm, num_classes=20).squeeze().permute(0,3,1,2).float()
            imgs = imgs.to(device, dtype=torch.float32)
            segm = segm.to(device, dtype=torch.float32)
            x = torch.cat((imgs, segm), dim=1)
            output_imgs, output_segs = model(x)
            
            for img, seg in zip(output_imgs, output_segs):
                img = denorm(img)
                
                save_image(img, f'output/{filename}/{str(img_count)}_img.png')

                if mode == 'complete':
                    #seg = seg.max(1)[1].cpu().numpy()[0]
                    seg = torch.argmax(seg, dim=0).cpu().numpy()
                    print(seg.shape)
                    #seg = Cityscapes.decode_target(seg)
                    seg = Image.fromarray(seg.astype('uint8'))

                    # new_img_dir = os.path.join(dataset_path, output_dir, img_path)
                    # os.makedirs(os.path.dirname(new_img_dir), exist_ok=True)
                    seg.save(f'output/{filename}/{str(img_count)}_seg.png') #TODO ricreare la cartella di output
                    break
                break
                
                img_count += 1

    print(f"Done!")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    example(img_path, checkpoint_path=checkpoint_path)
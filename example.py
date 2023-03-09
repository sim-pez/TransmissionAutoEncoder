import torch
from torchvision.transforms import ToTensor
import os
from PIL import Image
#from model import UNet
from datetime import datetime
from torchvision.utils import save_image
from torchvision import transforms

from model import SegmentationAutoencoder
from train import encoding_size


img_path = 'rightImg8bit_trainvaltest/rightImg8bit/test/berlin/berlin_000000_000019_rightImg8bit.png'
checkpoint_path = None #'checkpoints/2023-03-03 14:34:01.408481 enc-32/epoch199_[0.000045,0.000087].pth'

def example(img_path, checkpoint_path=None):
    '''
    Do an example usage of the model. The output will be saved in the output folder.
    If not checkpoint path is provided, the last checkpoint will be used.
    '''
    if not checkpoint_path:
        checkpoint_path = open('checkpoints/last_ckpt.txt', "r").read()

    transform = transforms.Compose([
                    transforms.Resize((256, 512)),
                    transforms.ToTensor()
                    ])
    
    model = SegmentationAutoencoder(encoding_size)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded from checkpoint ' + checkpoint_path)
    
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0) 
    with torch.no_grad():
        reconstructed_img, segmentation1, segmentation2 = model(image)

    segmentation1 = torch.argmax(segmentation1, dim=1).float()
    segmentation1 /= segmentation1.max(1, keepdim=True)[0]

    segmentation2 = torch.argmax(segmentation2, dim=1).float()
    segmentation2 /= segmentation2.max(1, keepdim=True)[0]
    
    img_name = 'output/{}_img_{}'.format(str(datetime.now()), os.path.basename(img_path))
    os.makedirs('output', exist_ok=True)
    save_image(reconstructed_img, img_name)

    seg1_name = 'output/{}_seg1_{}'.format(str(datetime.now()), os.path.basename(img_path))
    os.makedirs('output', exist_ok=True)
    save_image(segmentation1, seg1_name)

    seg2_name = 'output/{}_seg2_{}'.format(str(datetime.now()), os.path.basename(img_path))
    os.makedirs('output', exist_ok=True)
    save_image(segmentation2, seg2_name)


    print(f"Done!")


if __name__ == "__main__":
    example(img_path, checkpoint_path=checkpoint_path)
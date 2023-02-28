import torch
from torchvision.transforms import ToTensor
import os
from PIL import Image
from model import Autoencoder
from datetime import datetime
from torchvision.utils import save_image

from utils import find_device_and_batch_size
from train import encoding_size

img_path = 'rightImg8bit_trainvaltest/rightImg8bit/test/berlin/berlin_000000_000019_rightImg8bit.png'
checkpoint_path = None # "/Users/simone/Desktop/unit/autoencoder/checkpoints/2023-02-25 00:37:06.625761/epoch1.pth"

def example(img_path, checkpoint_path=None):
    '''
    Do an example usage of the model. The output will be saved in the output folder.
    If not checkpoint path is provided, the last checkpoint will be used.
    '''

    device, batch_size = find_device_and_batch_size()

    model = Autoencoder(encoding_size)
    if not checkpoint_path:
        checkpoint_path = open('checkpoints/last_ckpt.txt', "r").read()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded from checkpoint ' + checkpoint_path)

    transform = transforms.Compose([
                    transforms.Resize((1024, 2048)),
                    transforms.ToTensor()
                    ])
    
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    #model.to(device)
    #image.to(device)
    with torch.no_grad():
        reconstructed_image = model(image)

    reconstructed_image = reconstructed_image.cpu()
    save_image(reconstructed_image, 'output/{}_{}'.format(str(datetime.now()), os.path.basename(img_path)))


if __name__ == "__main__":
    example(img_path, checkpoint_path=checkpoint_path)
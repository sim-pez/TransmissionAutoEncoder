import torch
from torchvision.transforms import ToTensor
import os
from PIL import Image
from model import Autoencoder
from datetime import datetime

img_path = 'rightImg8bit_trainvaltest/rightImg8bit/test/berlin/berlin_000000_000019_rightImg8bit.png'
checkpoint_path = None # "/Users/simone/Desktop/unit/autoencoder/checkpoints/2023-02-25 00:37:06.625761/epoch1.pth"


def example(img_path, checkpoint_path=None):
    '''
    Do an example usage of the model. The output will be saved in the output folder.
    If not checkpoint path is provided, the last checkpoint will be used.
    '''

    model = Autoencoder()

    if not checkpoint_path:
        checkpoint_path = open('checkpoints/last_ckpt.txt', "r").read()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded from checkpoint ' + checkpoint_path)
    
    image = Image.open(img_path).convert("RGB") 
    image = ToTensor()(image)
    image = image.unsqueeze(0) 

    with torch.no_grad():
        reconstructed_image = model(image)

    reconstructed_image = reconstructed_image.squeeze(0)
    reconstructed_image = reconstructed_image.permute(1, 2, 0)
    reconstructed_image = reconstructed_image.numpy() 
    reconstructed_image = (reconstructed_image * 255).astype('uint8') 
    reconstructed_image = Image.fromarray(reconstructed_image)
    
    os.makedirs("output", exist_ok=True)
    reconstructed_image.save('output/{}_{}'.format(str(datetime.now()), os.path.basename(img_path)))
    print('Image reconstruction complete.')

if __name__ == "__main__":

    example(img_path, checkpoint_path=checkpoint_path)
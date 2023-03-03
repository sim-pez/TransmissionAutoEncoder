import torch
from torchvision.transforms import ToTensor
import os
from PIL import Image
from model import Autoencoder
from datetime import datetime
from torchvision.utils import save_image
from torchvision import transforms


from utils import find_device_and_batch_size, get_encoding_size

img_path = 'rightImg8bit_trainvaltest/rightImg8bit/test/berlin/berlin_000000_000019_rightImg8bit.png'
checkpoint_path = None #'checkpoints/2023-03-03 14:34:01.408481 enc-32/epoch199_[0.000045,0.000087].pth'

def example(img_path, checkpoint_path=None):
    '''
    Do an example usage of the model. The output will be saved in the output folder.
    If not checkpoint path is provided, the last checkpoint will be used.
    '''
    if not checkpoint_path:
        checkpoint_path = open('checkpoints/last_ckpt.txt', "r").read()

    device, batch_size = find_device_and_batch_size()
    encoding_size = get_encoding_size(checkpoint_path)
    
    model = Autoencoder(encoding_size)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded from checkpoint ' + checkpoint_path)

    transform = transforms.Compose([
                    transforms.Resize((256, 512)),
                    transforms.ToTensor()
                    ])
    
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0) 
    with torch.no_grad():
        reconstructed_image = model(image)

    reconstructed_image = reconstructed_image.cpu()
    reconstructed_image = reconstructed_image.squeeze(0)
    img_name = 'output/{}_{}'.format(str(datetime.now()), os.path.basename(img_path))
    os.makedirs('output', exist_ok=True)
    save_image(reconstructed_image, img_name)
    print(f"Done! saved {img_name}")


if __name__ == "__main__":
    example(img_path, checkpoint_path=checkpoint_path)
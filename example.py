import os
import torch
import warnings
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms

from model import SegmentationAutoencoder
from utils import get_parameters_from_checkpoint


img_path = 'rightImg8bit_trainvaltest/rightImg8bit/test/berlin/berlin_000000_000019_rightImg8bit.png'
checkpoint_path = 'checkpoints/mode:[complete]  enc:[32]  r:[0.8]/epoch:[039]  test:[0.14088]  train:[0.83827].pth'

def example(img_path, checkpoint_path):
    '''
    Do an example usage of the model. The output will be saved in the output folder.
    If not checkpoint path is provided, the last checkpoint will be used.
    '''


    #load model
    transform = transforms.Compose([
                    transforms.Resize((256, 512)),
                    transforms.ToTensor()
                    ])
    mode, encoding_size = get_parameters_from_checkpoint(checkpoint_path)
    model = SegmentationAutoencoder(mode=mode, encoding_size=encoding_size)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded from checkpoint ' + checkpoint_path)
    

    #load image
    image = Image.open(img_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0) 


    #predict
    with torch.no_grad():
        reconstructed_img, segmentation1, segmentation2 = model(image)

    if mode == 'segmentation_only' or mode == 'complete':
        segmentation1 = torch.argmax(segmentation1, dim=1).float()
        segmentation1 /= segmentation1.max(1, keepdim=True)[0]
    if mode == 'complete':
        segmentation2 = torch.argmax(segmentation2, dim=1).float()
        segmentation2 /= segmentation2.max(1, keepdim=True)[0]


    #save output
    os.makedirs('output', exist_ok=True)
    start_index = checkpoint_path.find('/') + 1
    end_index = checkpoint_path.find('/', start_index)
    filename = checkpoint_path[start_index:end_index]

    if mode == 'autoencoder_only' or mode == 'complete':
        save_image(reconstructed_img, f'output/{filename}_img.png')
    if mode == 'segmentation_only' or mode == 'complete':
        save_image(segmentation1, f'output/{filename}_seg1.png')
    if mode == 'complete':
        save_image(segmentation2, f'output/{filename}_seg2.png')


    print(f"Done!")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    example(img_path, checkpoint_path=checkpoint_path)
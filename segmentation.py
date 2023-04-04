'''
Create segmentated images inside dataset folder using resnet
'''
import os
import warnings
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm
from tqdm import tqdm

from PIL import Image
from torchvision import transforms as T

import resnet as rn
from dataset.cityscapes import Cityscapes
from utils import find_device_and_batch_size, find_images



def create_resnet_segmentation(dataset_path, colorized=False):

    torch.cuda.empty_cache()
    device, _ = find_device_and_batch_size()

    resnet = rn.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=19, output_stride=16)

    checkpoint = torch.load('checkpoints/mobilenet.pth', map_location=torch.device('cpu'))
    resnet.load_state_dict(checkpoint["model_state"])
    resnet = nn.DataParallel(resnet)
    if not force_cpu:
        resnet.to(device)
        print(f"using {device}")
    else:
        print(f"forcing cpu use")
    del checkpoint

    transform = T.Compose([
                T.Resize((256, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    if colorized:
        output_dir = 'resnet_seg_color'
    else:
        output_dir = 'resnet_seg_nocolor'
    os.makedirs(os.path.join(dataset_path, output_dir), exist_ok=True)

    with torch.no_grad():
        resnet = resnet.eval()
        input_path = os.path.join(dataset_path, 'leftImg8bit')

        for img_path in tqdm(find_images(input_path)):

            img_rel_path = os.path.join(input_path, img_path)
            img = Image.open(img_rel_path).convert('RGB')
            img = transform(img).unsqueeze(0)
            img = img.to(device)

            pred = resnet(img)

            pred = pred.max(1)[1].cpu().numpy()[0]
            if colorized:
                pred = Cityscapes.decode_target(pred)
            pred = Image.fromarray(pred.astype('uint8'))
            new_img_dir = os.path.join(dataset_path, output_dir, img_path)
            os.makedirs(os.path.dirname(new_img_dir), exist_ok=True)
            pred.save(new_img_dir)
    

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    dataset_path = 'dataset'
    create_resnet_segmentation(dataset_path)
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import map_values_tensor
import torch.nn.functional as F
import torch

from utils.utils import find_images, map_values_tensor, intmap



class ImageDataset(Dataset):

    def __init__(self, dataset_path, mode, get_img_path=False):
        self.dataset_path = dataset_path
        self.mode = mode
        self.get_img_path = get_img_path
        self.transform = transforms.Compose([transforms.Resize((256, 512)),
                                                 transforms.ToTensor()])
        self.images_folder = os.path.join(self.dataset_path, 'leftImg8bit', mode)
        self.image_list = find_images(self.images_folder)


    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label_path = image_path.replace("leftImg8bit", "resnet_seg_nocolor")
        label = Image.open(label_path)
        label = self.transform(label)
        label = map_values_tensor(label)
        label = F.one_hot(label, num_classes=20).permute(0,3,1,2).float().squeeze()

        if self.get_img_path:
            return image, label, self.image_list[idx]
        return image, label


    def __len__(self):
        return len(self.image_list)
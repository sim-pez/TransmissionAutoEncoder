import os
from PIL import Image
from torch.utils.data import Dataset

from utils import find_images

class ImageDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.image_list = find_images(dataset_folder)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_folder, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image

    def __len__(self):
        return len(self.image_list)
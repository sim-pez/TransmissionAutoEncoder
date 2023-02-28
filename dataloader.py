import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from utils import find_images

class ImageDataset(Dataset):

    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.transform = transforms.Compose([
                    transforms.Resize((1024, 2048)),
                    transforms.ToTensor()
                    ])
        self.image_list = find_images(dataset_folder)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_folder, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_list)
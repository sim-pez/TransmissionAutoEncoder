import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


from utils import find_images

class ImageDataset(Dataset):

    def __init__(self, images_folder, labels_folder):
        self.images_folder = images_folder
        self.labels_folder = labels_folder
        self.transform = transforms.Compose([
                    transforms.Resize((256, 512)),
                    transforms.ToTensor()
                    ])
        self.image_list = find_images(images_folder)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_folder, self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label_name = self.image_list[idx].replace("rightImg8bit", "gtFine_labelIds")
        label_path = os.path.join(self.labels_folder, label_name)
        label = Image.open(label_path) 
        image = self.transform(image)
        label = self.transform(label)
        
        return image, label

    def __len__(self):
        return len(self.image_list)
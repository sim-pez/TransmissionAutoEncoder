import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from model import Autoencoder
from dataloader import ImageDataset

num_epochs = 100
dataset_path = '/home/cr7036222/TransmissionAutoEncoder/rightImg8bit_trainvaltest'
load_from_checkpoint = True

def train(num_epochs, dataset_path, load_from_checkpoint=True):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("No GPU available, using the CPU instead.")

    transform = transforms.Compose([
                    transforms.Resize((2048, 1024)),
                    transforms.ToTensor()
                    ])
    train_set = ImageDataset(dataset_path, transform=transform)
    dataloader = DataLoader(train_set, batch_size=4, shuffle=True)

    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    if load_from_checkpoint:
        checkpoint_path = open('checkpoints/last_ckpt.txt', "r").read()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        checkpoint_dir = os.path.dirname(checkpoint_path)
        print('Model loaded from checkpoint ' + checkpoint_path)
    else:
        checkpoint_dir = "checkpoints/" + str(datetime.now())
        os.makedirs(checkpoint_dir, exist_ok=True)
        first_epoch = 0
        print('Model initialized from scratch')


    # Train the autoencoder
    for epoch in range(first_epoch, num_epochs):
        for data in tqdm(dataloader):
            img, _ = data
            # Forward pass
            output = model(img)
            loss = criterion(output, img)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        #save model
        checkpoint_path = os.path.join(checkpoint_dir, '[{}]_epoch{}.pth'.format(loss.item(), epoch+1))
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
        with open ('checkpoints/last_ckpt.txt', "w") as f:
            f.write(checkpoint_path)

if __name__ == "__main__":

    train(num_epochs, dataset_path, load_from_checkpoint)
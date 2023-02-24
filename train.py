import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from model import Autoencoder
from data_loader import ImageDataset

if __name__ == "__main__":

    num_epochs = 100
    dataset_path = '/Users/simone/Desktop/unit/autoencoder/rightImg8bit_trainvaltest/rightImg8bit/train'

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_dir = "checkpoints/" + str(datetime.now())
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train the autoencoder
    for epoch in range(num_epochs):
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
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch{}.pth'.format(epoch+1))
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
        with open ('checkpoints/last_ckpt.txt', "w") as f:
            f.write(checkpoint_path)
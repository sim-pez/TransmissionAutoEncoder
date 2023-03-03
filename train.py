import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from torchsummary import summary


from utils import find_device_and_batch_size
from model import Autoencoder
from dataloader import ImageDataset

load_from_checkpoint = False
force_cpu = False
num_epochs = 200
encoding_size = 16 #4, 16 or 32

dataset_path = 'rightImg8bit_trainvaltest/rightImg8bit'


def train(num_epochs, dataset_path, load_from_checkpoint=True):

    device, batch_size = find_device_and_batch_size()

    train_dataset = ImageDataset(os.path.join(dataset_path, "train"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ImageDataset(os.path.join(dataset_path, "test"))
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(encoding_size)
    if not force_cpu:
        model.to(device)
        print(f"Training with {device}")
    else:
        print(f"Training forcing cpu use")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    #summary(model, (3, 512, 256), device="cpu") 


    if load_from_checkpoint:
        checkpoint_path = open('checkpoints/last_ckpt.txt', "r").read()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        checkpoint_dir = os.path.dirname(checkpoint_path)
        print('Model loaded from checkpoint ' + checkpoint_path)
    else:
        checkpoint_dir = f"checkpoints/{str(datetime.now())} enc-{str(encoding_size)}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        first_epoch = 0
        print('Model initialized from scratch')


    # Train the autoencoder
    for epoch in range(first_epoch, num_epochs):
        
        train_loss = 0.0

        for imgs in tqdm(train_loader):
            if not force_cpu:
                imgs = imgs.to(device)
            output = model(imgs)
            loss = criterion(output, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss/len(train_dataset)

        test_loss = 0.0

        with torch.no_grad():
            for imgs in test_loader:
                if not force_cpu:
                    imgs = imgs.to(device)
                output = model(imgs)
                loss = criterion(output, imgs)
                test_loss += loss.item()

        avg_test_loss = test_loss/len(test_dataset)

        #save model
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch{}_[{:.6f},{:.6f}].pth'.format(epoch, avg_train_loss, avg_test_loss))
        torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, checkpoint_path)
        with open ('checkpoints/last_ckpt.txt', "w") as f:
            f.write(checkpoint_path)

        print('Epoch [{}/{}], Loss on train set: {:.6f}, Loss on test set: {:.6f}'.format(epoch+1, num_epochs, avg_train_loss, avg_test_loss))

if __name__ == "__main__":

    train(num_epochs, dataset_path, load_from_checkpoint)
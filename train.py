'''
Script to train the autoencoder and/or the segmentation network. 
It will generate a folder with the checkpoints if it doesn't exist.
Elsewhere, it will load the latest checkpoint and continue training from there.
'''

import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from model import SegmentationAutoencoder
from utils import find_device_and_batch_size, get_checkpoint_dir, get_last_checkpoint
from dataloader import ImageDataset
import warnings


encoding_size = 4   # 4, 16 or 32
r = 0.8             # image reconstruction rate
mode = 'segmentation_only'   # can be 'complete', 'segmentation_only', 'autoencoder_only'
lr = 0.005 #0.001
num_epochs = 200    # number of epochs to train
force_cpu = False   # force cpu use

img_set_path = 'rightImg8bit_trainvaltest/rightImg8bit' # path to the folder containing the images
label_set_path = 'gtFine_trainvaltest/gtFine'           # path to the folder containing the segmentation labels


def train(img_set_path, label_set_path, encoding_size, r, mode, lr, num_epochs, force_cpu=False):

    torch.cuda.empty_cache()
    device, batch_size = find_device_and_batch_size()

    train_dataset = ImageDataset(images_folder = os.path.join(img_set_path, "train"), 
                                 labels_folder = os.path.join(label_set_path, "train"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ImageDataset(images_folder = os.path.join(img_set_path, "val"),  # using validation set instead of test set
                                labels_folder= os.path.join(label_set_path, "val")) # 
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = SegmentationAutoencoder(mode=mode, encoding_size=encoding_size, r=r)

    if not force_cpu:
        model.to(device)
        print(f"Training with {device}")
    else:
        print(f"Training forcing cpu use")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    segm_criterion1 = torch.nn.CrossEntropyLoss()
    segm_criterion2 = torch.nn.CrossEntropyLoss()
    img_criterion = torch.nn.MSELoss()
    
    #summary(model, (3, 512, 256), device="cpu") 

    checkpoint_dir = get_checkpoint_dir(mode, encoding_size, r)
    if os.path.exists(checkpoint_dir):
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch'] + 1
        print('Checkpoint found. Model loaded from last one.')
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        first_epoch = 0
        print('Checkpoint not found. Model initialized from scratch')


    # Training loop
    for epoch in range(first_epoch, num_epochs):
        
        #train
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader):
            if not force_cpu:
                imgs = imgs.to(device)
                labels = labels.to(device)
            output, segmentation1, segmentation2 = model(imgs)

            if mode == 'complete':
                loss_img = img_criterion(output, imgs)
                loss_segm1 = segm_criterion1(segmentation1, labels)
                loss_segm2 = segm_criterion2(segmentation2, labels)
                total_loss = r * loss_img + ((1 - r) / 2) *  loss_segm1 + ((1 - r) / 2) * loss_segm2
            elif mode == 'segmentation_only':
                loss_segm1 = segm_criterion1(segmentation1, labels)
                total_loss = loss_segm1
            elif mode == 'autoencoder_only':
                loss_img = img_criterion(output, imgs)
                total_loss = loss_img

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        
        avg_train_loss = train_loss/len(train_dataset)


        #test
        test_loss = 0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                if not force_cpu:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                output, segmentation1, segmentation2 = model(imgs)

                if mode == 'complete':
                    loss_img = img_criterion(output, imgs)
                    loss_segm1 = segm_criterion1(segmentation1, labels)
                    loss_segm2 = segm_criterion2(segmentation2, labels)
                    total_loss = r * loss_img + ((1 - r) / 2) *  loss_segm1 + ((1 - r) / 2) * loss_segm2
                elif mode == 'segmentation_only':
                    loss_segm1 = segm_criterion1(segmentation1, labels)
                    total_loss = loss_segm1
                elif mode == 'autoencoder_only':
                    loss_img = img_criterion(output, imgs)
                    total_loss = loss_img
                    
                test_loss += total_loss.item()

        avg_test_loss = test_loss/len(test_dataset)
        

        #save model
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch:[{}]  test:[{:.5f}]  train:[{:.5f}].pth'.format(epoch, avg_train_loss, avg_test_loss))
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                    }, checkpoint_path)


        print('Epoch [{}/{}] end. Loss on train set: {:.5f}, Loss on test set: {:.5f}'.format(epoch, num_epochs, avg_train_loss, avg_test_loss))


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    train(img_set_path, label_set_path, encoding_size, r, mode, lr, num_epochs, force_cpu)






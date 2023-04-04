'''
Script to train the autoencoder. 
It will generate a folder with the checkpoints if it doesn't exist.
Elsewhere, it will load the latest checkpoint and continue training from there.
'''
import os
import warnings
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from utils import utils
import torch.utils.data as data
from dataset.cityscapes import Cityscapes
from dataloader import ImageDataset
from model import Autoencoder

from utils import find_device_and_batch_size
from utils.model_load_helpers import get_checkpoint_dir, get_last_checkpoint


encoding_size = 32   # 4, 8, 16 or 32
r = 0.8             # image reconstruction rate
mode = 'image_only'   # can be 'complete', 'image_only'
lr = 0.02
num_epochs = 70    # number of epochs to train
force_cpu = False   # force cpu use


def train(dataset_path, encoding_size, r, mode, lr, num_epochs, force_cpu = False):

    torch.cuda.empty_cache()
    device, batch_size = find_device_and_batch_size()
    if force_cpu:
        device = torch.device('cpu')
        print(f"Forced CPU use")
    print(f"Training with {device}")


    train_dataset = ImageDataset(dataset_path=dataset_path, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    test_dataset = ImageDataset(dataset_path=dataset_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    model = Autoencoder(mode=mode, encoding_size=encoding_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    segm_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    img_criterion = torch.nn.MSELoss()
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=num_epochs)

    checkpoint_dir = get_checkpoint_dir(mode, encoding_size, r)
    if os.path.exists(checkpoint_dir):
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            first_epoch = checkpoint['epoch'] + 1
            print('Checkpoint found. Model loaded from last one.')
            del checkpoint
        else:
            first_epoch = 0
            print('No valid checkpoint found. Model initialized from scratch.')
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        first_epoch = 0
        print('Checkpoint directory not found. Model initialized from scratch.')


    # Training loop
    for epoch in range(first_epoch, num_epochs):
        
        #train
        train_loss = 0.0
        for imgs, segm in tqdm(train_loader):
            imgs = imgs.to(device)#(device, dtype=torch.float32)
            segm = segm.to(device)#(device, dtype=torch.float32)
            x = torch.cat((imgs, segm), dim=1)
            output_img, output_seg = model(x)
            if mode == 'complete':
                loss_img = img_criterion(output_img, imgs)
                loss_segm = segm_criterion(output_seg, segm)
                batch_loss = r * loss_img + (1 - r) * loss_segm
            elif mode == 'image_only':
                batch_loss = img_criterion(output_img, imgs)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
        
        scheduler.step()

        avg_train_loss = train_loss/len(train_dataset)


        # test
        test_loss = 0.0
        with torch.no_grad():
            for imgs, segm in test_loader:
                imgs = imgs.to(device)#(device, dtype=torch.float32)
                segm = segm.to(device)#(device, dtype=torch.float32)
                x = torch.cat((imgs, segm), dim=1)
                output_img, output_seg = model(x)

                if mode == 'complete':
                    loss_img = img_criterion(output_img, imgs)
                    loss_segm = segm_criterion(output_seg, segm)
                    batch_loss = r * loss_img + (1 - r) * loss_segm
                elif mode == 'image_only':
                    batch_loss = img_criterion(output_img, imgs)
                    
                test_loss += batch_loss.item()

        avg_test_loss = test_loss/len(test_dataset)
        

        #save model
        checkpoint_path = os.path.join(checkpoint_dir, 'epoch:[{}]  test:[{:.5f}]  train:[{:.5f}].pth'.format(str(epoch).zfill(3), avg_train_loss, avg_test_loss))
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'scheduler': scheduler.state_dict()
                    }, checkpoint_path)

        print('Epoch [{}/{}] end. Loss on train set: {:.5f}, Loss on test set: {:.5f}, lr: {:.10f}'.format(epoch, num_epochs, avg_train_loss, avg_test_loss, optimizer.param_groups[0]["lr"]))


if __name__ == "__main__":

    import torch
    from PIL import Image
    import torchvision.transforms as transforms

    warnings.filterwarnings('ignore')
    dataset_path = 'dataset'

    train(dataset_path, encoding_size, r, mode, lr, num_epochs, force_cpu)
'''
Script to train the autoencoder. 
It will generate a folder with the checkpoints if it doesn't exist.
Elsewhere, it will load the latest checkpoint and continue training from there.
'''
import os
import torch
import warnings
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from dataloader import ImageDataset
from model import Autoencoder

from utils import find_device_and_batch_size
from utils.model_load_helpers import get_checkpoint_dir, get_last_checkpoint



def get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='dataset',
                        help="Path to the dataset directory. If not specified, it will use the 'dataset' folder")
    parser.add_argument("--model", type=str, default='complete',
                        choices=['complete', 'image_only'], help='Model type')
    parser.add_argument("--lr", type=float, required=False, default=0.02, help='learning rate')
    parser.add_argument("--r", type=float, required=False, default=0.8, help="Image reconstruction rate")
    parser.add_argument("--c", type=int, required=False, default=32, help='Encoding channels. Bigger value has lower compression rate')
    parser.add_argument("--epochs", type=int, required=False, default=100, help='Number of epochs to train')

    return parser


def train(dataset_path, encoding_size, r, mode, lr, num_epochs):

    torch.cuda.empty_cache()
    device, batch_size = find_device_and_batch_size()
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
            imgs = imgs.to(device)
            segm = segm.to(device)
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
                imgs = imgs.to(device)
                segm = segm.to(device)
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

    print('Ended training')

if __name__ == "__main__":

    opts = get_argparser().parse_args()
    train(opts.dataset, opts.c, opts.r, opts.model, opts.lr, opts.epochs)

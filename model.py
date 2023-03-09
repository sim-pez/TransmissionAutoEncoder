import torch
from torch import nn

import segmentation_models_pytorch as smp


class SegmentationAutoencoder(nn.Module):
    def __init__(self, encoding_size):
        super(SegmentationAutoencoder, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(38, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, encoding_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoding_size, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        self.segmentator1 = smp.Unet('efficientnet-b0', classes=35, activation='softmax')
        self.segmentator2 = smp.Unet('efficientnet-b0', classes=35, activation='softmax')

    def forward(self, x):
        segmentation1 = self.segmentator1(x)
        x = torch.cat((x, segmentation1), dim=1) #TODO refactor
        x = self.encoder(x) # embedding size: torch.Size([batch_num, encoding_size, 32, 64])
        x = self.decoder(x)
        segmentation2 = self.segmentator2(x)
        return x, segmentation1, segmentation2
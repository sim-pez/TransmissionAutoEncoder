import torch
from torch import nn

import segmentation_models_pytorch as smp


class SegmentationAutoencoder(nn.Module):
    def __init__(self, encoding_size, mode = 'complete'):
        super(SegmentationAutoencoder, self).__init__()

        mode_types = ['complete', 'segmentation_only', 'autoencoder_only']
        if mode not in mode_types:
            raise ValueError("Invalid mode type. Expected one of: %s" % mode_types)

        self.mode = mode

        if mode != 'segmentation_only'
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
        
        if mode != 'autoencoder_only'
            self.segmentator1 = smp.Unet('efficientnet-b0', classes=35, activation='softmax')
        if mode == 'complete'
            self.segmentator2 = smp.Unet('efficientnet-b0', classes=35, activation='softmax')

    def forward(self, x):
        if self.mode == 'complete':
            segmentation1 = self.segmentator1(x)
            x = torch.cat((x, segmentation1), dim=1)
            x = self.encoder(x)
            x = self.decoder(x)
            segmentation2 = self.segmentator2(x)
            return x, segmentation1, segmentation2

        elif self.mode == 'autoencoder_only'
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
        elif self.mode == 'segmentation_only'
            x = self.segmentator1(x)
        
        else:
            raise ValueError("Something went wrong with mode types!")
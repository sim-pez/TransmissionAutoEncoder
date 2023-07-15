import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, mode, encoding_size):
        super(Autoencoder, self).__init__()

        #check parameters
        self.mode = mode
        mode_types = ['complete', 'image_only']
        if mode not in mode_types:
            raise ValueError("Invalid mode type. Expected one of: %s" % mode_types)
        elif mode == 'image_only':
            concatenation_channels = 3
        elif mode == 'complete':
            concatenation_channels = 3 + 20
        
        
        self.encoder = nn.Sequential( 
            nn.Conv2d(concatenation_channels, 64, kernel_size=3, padding=1),
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
            nn.ConvTranspose2d(64, concatenation_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        
        if self.mode == 'complete':
            x = self.encoder(x)
            x = self.decoder(x)
            img, segmentation = torch.split(x, [3, 20], dim=1)
            return img, segmentation
            
        elif self.mode == 'image_only':
            x, _ = torch.split(x, [3, 20], dim=1)
            x = self.encoder(x)
            x = self.decoder(x)
            return x, None
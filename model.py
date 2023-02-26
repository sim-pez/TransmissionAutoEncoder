import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoding_size):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        # Encoding layer
        self.encoding = nn.Sequential(
            nn.Linear(256 * 128 * 64, encoding_size),
            nn.ReLU(),
        )

        # Decoding layer
        self.decoding = nn.Sequential(
            nn.Linear(encoding_size, 256 * 128 * 64),
            nn.ReLU(),
        )

    def forward(self, x):
        # Encode the input
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        encoding = self.encoding(x)

        # Decode the encoding
        x = self.decoding(encoding)
        x = x.view(x.size(0), 256, 128, 64)
        x = self.decoder(x)

        return x

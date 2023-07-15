# TransmissionAutoEncoder

This architecture consists of a ResNet followed by an Autoencoder. The goal is to compress both the image and its segmentation and transmit the resulting embeddings in the latent space through a mobile network. Additionally, there is a parameter that allows for selecting the size of the embeddings, which simulates a bottleneck in the network

<p float="left" align="center">
  <img src="docs/architecture.png" width="100%"  />
</p>


# Usage
## Install
```
conda create --name autoenc --file requirements.txt
```

## Example script for an image
```
python3 example.py
```

## Train

1. Download the dataset from [here](https://www.cityscapes-dataset.com/downloads/) (free registration needed). I reccomend to use `leftImg8bit_trainvaltest.zip` `gtFine_trainvaltest.zip`.
2. Run `python3 train.py`


# TransmissionAutoEncoder

This Neural Network is basically an autoencoder with a [U-Net](https://en.wikipedia.org/wiki/U-Net). The objective is to send through a mobile network the embedding of the image itself and its segmentation. There is also a parameter to choose the size of embeddings to simulate the network bottleneck

# Usage
## Install

Run
```
conda create --name autoenc --file requirements.txt
```

## Example script for an image

Run
```
python3 example.py
```

## Train

1. Download the dataset from [here](https://www.cityscapes-dataset.com/downloads/) (free registration needed). I reccomend to use `leftImg8bit_trainvaltest.zip` `gtFine_trainvaltest.zip`.
2. Run `python3 train.py`


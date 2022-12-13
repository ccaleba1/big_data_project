import matplotlib
matplotlib.use('Agg')
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data
import torchvision.datasets as dset
from loader import CustomDataset
from cnn import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = 64
batch_size = 75
workers = 3
dataroot = '~/git/big_data_project/data/' #location of data directory HERE




train_dataset = CustomDataset("data/1219/labels.csv", "data/1219/rgb")

image, label = train_dataset[0]

print(image, label)

num_epochs = 30
lr = 0.0002
beta1 = 0.5
nz = 100

cnn = CNN().to(device)

crit = nn.BCELoss()


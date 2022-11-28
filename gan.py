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
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from models import Generator
from models import Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

with open('data_array.pkl', 'rb') as f:
    train = pickle.load(f)

epochs = 10
lr = 0.0002
beta1 = 0.5

train = torch.tensor(train).to(device)
train = torch.reshape(train, (train.shape[0], 3, 240, 320)).float()

gen = Generator().float().to(device)
disc = Discriminator().float().to(device)

gen.apply(weights_init)
disc.apply(weights_init)

crit = nn.BCELoss()
noise = torch.randn(64, 3, 1, 1)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in tqdm(range(0,epochs)):
    disc.zero_grad()

    output = disc(train)
    label = torch.ones(*output.shape).float()

    errD_real = crit(output.float(), label.float())
    errD_real.backward()
    D_x = output.mean().item()

    noise = torch.randn(train.shape)
    fake = gen(noise)
    label.fill_(fake_label)

    output = disc(fake.detach())
    errD_fake = crit(output, label)













# grid_img = torchvision.utils.make_grid(train[:10], nrow=5)
#
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()
#
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[:64], padding=2, normalize=True),(1,2,0)))
# plt.show()
# epochs = 10
#1301, 240, 320, 3

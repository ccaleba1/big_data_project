import os
import random
import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

data = []
#for filename in os.listdir("data/1301/rgb"):

#    file = "data/1301/rgb/" + str(filename)
#    img = Image.open(file)
#    data_array = np.array(img)
#    data.append(data_array)

#data = np.array(data)
ngpu = 2
image_size = 64
batch_size = 125
workers = 3
dataroot = '~/big_data_project/data/1301/'

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
#real_batch = next(iter(dataloader))

#data = np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0))
#plt.imsave('test.png', data)

results = torch.tensor(dataloader)
with open('data_array.pkl', 'wb') as f:
    pickle.dump(results, f)

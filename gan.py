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
from models import Generator
from models import Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ngpu = 2
image_size = 64
batch_size = 75
workers = 3
dataroot = '~/git/big_data_project/data/' #location of data directory HERE

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
real_batch = next(iter(dataloader))

num_epochs = 30
lr = 0.0002
beta1 = 0.5
nz = 100

gen = Generator().float().to(device)
disc = Discriminator().float().to(device)

gen.apply(weights_init)
disc.apply(weights_init)

crit = nn.BCELoss()
const_noise = torch.randn(image_size, nz, 1, 1, device=device)
real_label = 1.0
fake_label = 0.0

#optimizerD = optim.SGD(disc.parameters(), lr=0.1, momentum=0.9)
optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        disc.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = disc(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = crit(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = gen(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disc(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = crit(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        gen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disc(fake).view(-1)
        # Calculate G's loss based on this output
        errG = crit(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or (epoch == num_epochs-1):
            with torch.no_grad():
                fake2 = gen(const_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake2.detach().cpu(), padding=2, normalize=True))
        iters += 1

print("Saving Generator Model...")
torch.save(gen, "generator.tensor")

print("Saving Generator/Discriminator Loss Plot...")
fig = plt.figure(figsize=(10,5))
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Generator_Discriminator_Loss.png')
plt.close(fig)
print("Done")

print("Saving Generator Output...")
real_batch = next(iter(dataloader))

# Plot the real images
fig1 = plt.figure()
plt.axis("off")
plt.title("Real Images")
img1 = np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0))
plt.imsave("real_images.png", img1.cpu().numpy())
plt.close(fig1)

# Plot the fake images from the last epoch
fig2 = plt.figure()
plt.axis("off")
plt.title("Fake Images")
img2 = np.transpose(img_list[-1],(1,2,0))
plt.imsave("fake_images.png", img2.cpu().numpy())
plt.close(fig2)
print("Done!")

# %%
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py

import os
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm
# %% Parameters
n_epochs = 50
batch_size = 4
lr = .0002
b1 = 0.5
b2 = .999

latent_dim = 10
img_size = 28
channels = 1
sample_interval = 400

img_shape = (channels, img_size, img_size)
# %% Load data.
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root = '../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                     batch_size=64,
                                     shuffle=True)
# %%
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim)))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
    
# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Tensor(np.random.normal(0, 1, (n_row ** 2, latent_dim)))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
# %%
for epoch in range(n_epochs):
    print(f'Epoch {epoch}/{n_epochs}')
    for i, (imgs, _) in tqdm(enumerate(data_loader)):

        # Adversarial ground truths
        valid = Tensor(imgs.shape[0], 1).fill_(1.0)
        fake =  Tensor(imgs.shape[0], 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim)))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, n_epochs, i, len(data_loader), d_loss.item(), g_loss.item())
        # )

        batches_done = epoch * len(data_loader) + i
        if batches_done % sample_interval == 0:
            pass
            # sample_image(n_row=10, batches_done=batches_done)

# %% Get encodings after given number of epochs
finalEncodings = []
allLabels = []
for imgs, labels in tqdm(data_loader):
    real_imgs = imgs.type(Tensor)
    
    encodings = encoder(real_imgs).detach().cpu().numpy()
    finalEncodings.append(encodings)
    allLabels.append(labels.detach().cpu().numpy())

allLabels = np.concatenate(allLabels)
finalEncodings = np.concatenate(finalEncodings)
# %% Look at some reconstructions
for imgs, labels in tqdm(data_loader):
    real_imgs = imgs.type(Tensor)
    
    encodings = encoder(real_imgs)
    decodings = decoder(encodings)
    break

img0 = imgs[0][0].detach().cpu().numpy()
label0 = labels[0].detach().cpu().numpy()
decoding0 = decodings[0][0].detach().cpu().numpy()
# %%
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto', 
                  init='random', perplexity=3).fit_transform(finalEncodings)
# %%
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=allLabels, cmap='tab10')
plt.colorbar()
# %% TSNE UMAP
c = 0
tsneImgs = []
tsneLabels = []
for imgs, labels in tqdm(data_loader):
    for img in imgs:
        img = img.detach().cpu().numpy().ravel()
        tsneImgs.append(img)

        c +=1
    tsneLabels.append(labels.detach().cpu().numpy())
    if c >= 10000:
        break

tsneLabels = np.concatenate(tsneLabels)
tsneImgs = np.array(tsneImgs)
# %%
X_embedded_full = TSNE(n_components=2, learning_rate='auto', 
                  init='random', perplexity=3).fit_transform(tsneImgs)
# %%
plt.scatter(X_embedded_full[:,0], X_embedded_full[:,1], c=tsneLabels, cmap='tab10')
plt.colorbar()
# %%
import matplotlib.pyplot as plt
import re
# %%
with open('./aaeRes.txt') as outFile:
    x = outFile.read()
x = x.split('\n')

GLoss, DLoss = [], []
for line in x:
    losses = re.findall("\d+\.\d+", line)
    if len(losses) == 2:
        DLoss.append(float(losses[0]))
        GLoss.append(float(losses[1]))

# %%
plt.subplot(121)
plt.plot(DLoss)
plt.subplot(122)
plt.plot(GLoss)
# %%

# %%
from model import BetaVAE_H, BetaVAE_B
from solver import reconstruction_loss, kl_divergence
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.nn import functional as F

import numpy as np
from tqdm import tqdm
# %% Model parameters
train = True
seed = 1
cuda = True
max_iter = 1e6
batch_size = 64
z_dim = 10
beta = 4
objective = 'H'
model = 'H'
gamma = 1000
C_max = 25
C_stop_iter = 1e5
lr = 1e
beta1 = 0
beta2 = 0


decoder_dist = 'gaussian'
# %%
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
cifarData = datasets.CIFAR10(root = '../data', train=True, download=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_loader = torch.utils.data.DataLoader(dataset=cifarData,
                                     batch_size=batch_size,
                                     shuffle=True)
# %%
net = BetaVAE_H
# %%
global_iter = 0
for x in data_loader:
    global_iter += 1

    x_recon, mu, logvar = net(x)
    recon_loss = reconstruction_loss(x, x_recon, decoder_dist)
    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

    if objective == 'H':
        beta_vae_loss = recon_loss + beta*total_kld
    elif objective == 'B':
        C = torch.clamp(C_max/C_stop_iter*global_iter, 0, C_max.data[0])
        beta_vae_loss = recon_loss + gamma*(total_kld-C).abs()
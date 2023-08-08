# %%
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

# %%
def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, mean=1.0, std=0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)

# %% Inputs
n_epochs = 10
batch_size = 4
lr = .0002
b1 = 0.5
b2 = .999

img_size = 217

scale    = int(8 / (512 / img_size))
nLatentDims     = 56
nChIn           = 1
nChOut          = 1
dropoutRate     = 0.2
# %%
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root = '../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                     batch_size=64,
                                     shuffle=True)

# %%
dataiter = iter(data_loader)
images, labels = next(dataiter)

images = transforms.Resize(120)(images).cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.step1   = nn.Conv2d(nChIn, 64, 4, 2, 1)
        self.step2   = nn.BatchNorm2d(64)
        self.step3   = nn.PReLU()

        self.step4   = nn.Conv2d(64, 128, 4, 2, 1)
        self.step5   = nn.BatchNorm2d(128)
        self.step6   = nn.PReLU()

        self.step7   = nn.Conv2d(128, 256, 4, 2, 1)
        self.step8   = nn.BatchNorm2d(256)
        self.step9   = nn.PReLU()

        self.step10  = nn.Conv2d(256, 512, 4, 2, 1)
        self.step11  = nn.BatchNorm2d(512)
        self.step12  = nn.PReLU()

        self.step13  = nn.Conv2d(512, 1024, 4, 2, 1)
        self.step14  = nn.BatchNorm2d(1024)
        self.step15  = nn.PReLU()

        self.step16  = nn.Conv2d(1024, 1024, 4, 2, 1)
        self.step17  = nn.BatchNorm2d(1024)

        self.step18  = nn.Flatten()
        self.step19  = nn.PReLU()
        
        self.step20  = nn.Linear(1024, nLatentDims)
        self.step21  = nn.BatchNorm1d(nLatentDims)
    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        
        x = self.step4(x)
        x = self.step5(x)
        x = self.step6(x)
        
        x = self.step7(x)
        x = self.step8(x)
        x = self.step9(x)
        
        x = self.step10(x)
        x = self.step11(x)
        x = self.step12(x)
        
        x = self.step13(x)
        x = self.step14(x)
        x = self.step15(x)
        
        x = self.step16(x)
        x = self.step17(x)
        
        x = self.step18(x)
        x = self.step19(x)
        
        x = self.step20(x)
        x = self.step21(x)
        return x
    
encoder = Encoder().to(device)

outEncode = encoder(images)  
# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.step1  = nn.Linear(nLatentDims, 1024)
        self.step2  = nn.Unflatten(1, (1024, 1, 1))
        self.step3  = nn.PReLU()

        self.step4  = nn.ConvTranspose2d(1024, 1024, 4, 2, 1, 1)
        self.step5  = nn.BatchNorm2d(1024)
        self.step6  = nn.PReLU()

        self.step7  = nn.ConvTranspose2d(1024, 512, 4, 2, 1, 1)
        self.step8  = nn.BatchNorm2d(512)
        self.step9  = nn.PReLU()

        self.step10 = nn.ConvTranspose2d(512, 256, 4, 2, 1, 1)
        self.step11 = nn.BatchNorm2d(256)
        self.step12 = nn.PReLU()

        self.step13 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.step14 = nn.BatchNorm2d(128)
        self.step15 = nn.PReLU()

        self.step16 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.step17 = nn.BatchNorm2d(64)
        self.step18 = nn.PReLU()
        
        self.step19 = nn.ConvTranspose2d(64, 1, 4, 2, 1)
        self.step20 = nn.Sigmoid()

    def forward(self, x):
        
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        
        x = self.step4(x)
        x = self.step5(x)
        x = self.step6(x)
        
        x = self.step7(x)
        x = self.step8(x)
        x = self.step9(x)
        
        x = self.step10(x)
        x = self.step11(x)
        x = self.step12(x)
        
        x = self.step13(x)
        x = self.step14(x)
        x = self.step15(x)
        
        x = self.step16(x)
        x = self.step17(x)
        x = self.step18(x)
        
        x = self.step19(x)
        x = self.step20(x)
        
        return x
decoder = Decoder().to(device)

outDecode = decoder(outEncode)
# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.step1 = nn.Linear(nLatentDims, 1024)
        self.step2 = nn.LeakyReLU(0.2, inplace=True)
        self.step3 = nn.Linear(1024, 1024)
        self.step4 = nn.BatchNorm1d(1024)
        self.step5 = nn.LeakyReLU(0.2, inplace=True)
        self.step6 = nn.Linear(1024, 512)
        self.step7 = nn.BatchNorm1d(512)
        self.step8 = nn.LeakyReLU(0.2, inplace=True)
        self.step9 = nn.Linear(512, 1)
        self.step10 = nn.Sigmoid()
    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        x = self.step5(x)
        x = self.step6(x)
        x = self.step7(x)
        x = self.step8(x)
        x = self.step9(x)
        x = self.step10(x)
        return(x)
discriminator = Discriminator().to(device)
outDiscrim = discriminator(outEncode)
# %%
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()
# %%
for epoch in range(n_epochs):
    print(f'Epoch {epoch}/{n_epochs}')
    for i, (imgs, _) in tqdm(enumerate(data_loader)):
        imgs = transforms.Resize(120)(images).cuda()
        imgs = imgs.to(device)
        # Adversarial ground truths
        valid =  torch.Tensor(imgs.shape[0], 1).fill_(1.0).to(device)
        fake  =  torch.Tensor(imgs.shape[0], 1).fill_(0.0).to(device)


        # Configure input
        real_imgs = imgs

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(decoded_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], nLatentDims)), dtype=torch.float32).to(device)

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
        # if batches_done % sample_interval == 0:
        #     pass
        #     # sample_image(n_row=10, batches_done=batches_done)
# %% Look at some reconstructions
encodings, labels = [], []
c = 0
for imgs, label in tqdm(data_loader):
    imgs = transforms.Resize(120)(images).cuda()
    imgs = imgs.to(device)    
    encoding = encoder(imgs)
    decoding = decoder(encoding)
    encodings.append(encoding.detach().cpu().numpy())
    labels.append(label.detach().cpu().numpy())
    c +=1
    if c > 25:
        break
encodings = np.concatenate(encodings)
labels = np.concatenate(labels)
img0 = imgs[0][0].detach().cpu().numpy()
label0 = label[0].detach().cpu().numpy()
decoding0 = decoding[0][0].detach().cpu().numpy()

plt.subplot(121)
plt.imshow(img0)
plt.subplot(122)
plt.imshow(decoding0)
# %%
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto', 
                  init='random', perplexity=3).fit_transform(encodings)
# %%
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap='tab10')
plt.colorbar()
# %%

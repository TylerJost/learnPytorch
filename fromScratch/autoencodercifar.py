# %%
import torch
import torch.nn as nn

import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
# %%
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
cifarData = datasets.CIFAR10(root = '../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=cifarData,
                                     batch_size=64,
                                     shuffle=True)
# %%
dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))

# %%
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # nimages, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),# N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),# N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels=32, kernel_size=7), # N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels = 16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encoding(self, x):
        return self.encoder(x)
    
# Note: nn.MaxPool2d -> nn.MaxUnpool2d will undo this, but different strides, paddings, etc 
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNNAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)
model.cuda()
model.to('cuda')
# %%
num_epochs = 25
outputs = []
allLoss = []
for epoch in range(num_epochs):
    for (img, label) in tqdm(data_loader):
        img = img[:, 0:1, :, :]
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# %%
idx = 33
images = images[:, 0:1, :, :]
image = images[idx]
imgOut = model(image.to(device))
imageRecon = imgOut.cpu().detach().numpy()[0]

plt.subplot(121)
plt.imshow(image[0].numpy())
plt.subplot(122)
plt.imshow(imageRecon)
# %%

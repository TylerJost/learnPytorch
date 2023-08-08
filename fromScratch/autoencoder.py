# %%
import torch
import torch.nn as nn

import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm
# %%
transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root = '../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                     batch_size=64,
                                     shuffle=True)
# %%
dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))
# %%
class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        # nimages, 784 (28*28)
        self.encoder = nn.Sequential(
            nn.Linear(in_features = 28*28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=latent_dims)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features = latent_dims, out_features=12),
            nn.ReLU(),
            nn.Linear(in_features=12, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encoding(self, x):
        return self.encoder(x)
# Note: Keep last layer in mind. The range of images is [0, 1], which is why we use 
# the sigmoid activation function. 
# For example, for an image between [-1, 1] we would want a tanh activation function
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Autoencoder(20).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)
model.cuda()
model.to('cuda')
# %%
num_epochs = 15
outputs = []
allLoss = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    for (img, label) in tqdm(data_loader):
        img = img.reshape(-1, 28*28)
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

allEncodings, allLabels = [], []
for (img, label) in tqdm(data_loader):
    img = img.reshape(-1, 28*28)
    img = img.to(device)
    encoding = model.encoding(img)
    allEncodings.append(encoding.to('cpu').detach().numpy())
    allLabels.append(label.to('cpu').detach().numpy())

allEncodings = np.concatenate(allEncodings)
allLabels = np.concatenate(allLabels)
# %%
from sklearn.manifold import TSNE
X = allEncodings
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
# %%
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=allLabels, cmap='tab10')
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, allLabels, test_size=0.33, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=9000).fit(X_train, y_train)
y_pred = clf.predict(X_test)
# %%
idx = 2
image = images[idx]
imgRe = images.reshape(-1, 28*28)
imgOut = model(imgRe.to(device))
imgOut = imgOut.reshape(-1, 28, 28)
imageRecon = imgOut[idx].cpu().detach().numpy()

plt.subplot(121)
plt.imshow(image[0].numpy())
plt.subplot(122)
plt.imshow(imageRecon)      
# %%
encodings = []
labels = []

for (img, label) in tqdm(data_loader):
    img = img.reshape(-1, 28*28)
    img = img.to(device)
    encode = model.encoding(img.to(device)).cpu().detach().numpy()
    encodings.append(encode)
    labels.append(label.cpu().detach().numpy())

encodings = np.concatenate(encodings)
labels = np.concatenate(labels)
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
num_epochs = 1
outputs = []
allLoss = []
for epoch in range(num_epochs):
    for (img, label) in tqdm(data_loader):
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# %%
idx = 32
image = images[idx]
imgOut = model(image.to(device))
imageRecon = imgOut.cpu().detach().numpy()[0]

plt.subplot(121)
plt.imshow(image[0].numpy())
plt.subplot(122)
plt.imshow(imageRecon)

# Exercise: use maxpool2d
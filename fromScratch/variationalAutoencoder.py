# %%
# Comes from this tutorial: https://avandekleut.github.io/vae/
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.manifold import TSNE
# %% Get data like in autoencoder
transform =  torchvision.transforms.ToTensor()
mnist_data = torchvision.datasets.MNIST(root = '../data', train=True, download=True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                     batch_size=64,
                                     shuffle=True)
# %% Build model
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VariationalAutoencoder(2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)
model.cuda()
model.to('cuda')
# %%
num_epochs = 15
outputs = []
allLoss = []
criterion = lambda x, xhat: ((x - xhat)**2).sum() + model.encoder.kl

for epoch in range(num_epochs):
    for (img, label) in tqdm(data_loader):
        # img = img.reshape(-1, 28*28)
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    allLoss.append(float(loss))
# %% Get plotting of latent dimensions of 2
allLabels, allEncoding = [], []
for (img, label) in tqdm(data_loader):
    img = img.to(device)
    encoding = model.encoder(img).to('cpu').detach().numpy()
    allEncoding.append(encoding)
    allLabels.append(label.to('cpu').detach().numpy())
allEncoding = np.concatenate(allEncoding)
allLabels = np.concatenate(allLabels)
# %%
plt.scatter(allEncoding[:,0], allEncoding[:,1], c=allLabels, cmap='tab10')
plt.colorbar()
# %%
latentDims = [2, 10, 60, 100]
latentDimsRes = {dim: {'model': [], 'loss': []} for dim in latentDims}

for dim in latentDims:
    print(f'Dim: {dim} \n')
    model = VariationalAutoencoder(dim).to(device)
    model.cuda()
    model.to('cuda')
    for epoch in range(num_epochs):
        for (img, label) in tqdm(data_loader):
            # img = img.reshape(-1, 28*28)
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        latentDimsRes[dim]['loss'].append(float(loss))
    latentDimsRes[dim]['model'] = model

# %% Plot losses
minLosses = [min(latentDimsRes[dim]['loss']) for dim in latentDims]
# %% Get encodings
for dim in latentDims:
    latentDimsRes[dim]['encodings'] = []
allLabels = []
for (img, label) in tqdm(data_loader):
    img = img.to(device)
    for dim in latentDims:
        model = latentDimsRes[dim]['model']

        encoding = model.encoder(img).to('cpu').detach().numpy()
        latentDimsRes[dim]['encodings'].append(encoding)
    allLabels.append(label.to('cpu').detach().numpy())

allLabels = np.concatenate(allLabels)
for dim in latentDims:
    latentDimsRes[dim]['encodings'] = np.concatenate(latentDimsRes[dim]['encodings'])
# %%
X = latentDimsRes[10]['encodings']
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(X)
# %%
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=allLabels)
# %%

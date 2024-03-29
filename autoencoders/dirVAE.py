# %%
# Taken from https://github.com/is0383kk/Dirichlet-VAE/blob/main/dir_vae.py

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
# %%
batch_size = 128
epochs = 10
seed = 1234
log_interval = 1000
category = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)
# %%
ngf = 64
ndf = 64
nc = 1

def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    # ラプラス近似で正規分布に近似
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # Parameters of prior distribution after approximation

class Dir_VAE(nn.Module):
    def __init__(self):
        super(Dir_VAE, self).__init__()
        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
            # F.max_pool2d(input, kernel_size = input.size()[2:])
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, category)
        self.fc22 = nn.Linear(512, category)

        self.fc3 = nn.Linear(category, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

        # Dir prior
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(category, 0.3)) # 0.3 is a hyper param of Dirichlet distribution
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False


    def encode(self, x):
        conv = self.encoder(x);
        h1 = self.fc1(conv.view(-1, 1024))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, gauss_z):
        dir_z = F.softmax(gauss_z,dim=1) 
        # This variable (z) can be treated as a variable that follows a Dirichlet distribution (a variable that can be interpreted as a probability that the sum is 1)
        # Use the Softmax function to satisfy the simplex constraint
        # シンプレックス制約を満たすようにソフトマックス関数を使用
        h3 = self.relu(self.fc3(dir_z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view(-1,1024,1,1)
        return self.decoder(deconv_input)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        gauss_z = self.reparameterize(mu, logvar) 
        # gause_z is a variable that follows a multivariate normal distribution
        # Inputting gause_z into softmax func yields a random variable that follows a Dirichlet distribution (Softmax func are used in decoder)
        dir_z = F.softmax(gauss_z,dim=1) # This variable follows a Dirichlet distribution
        return self.decode(gauss_z), mu, logvar, gauss_z, dir_z

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, K):
        beta = 1.0
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        # ディリクレ事前分布と変分事後分布とのKLを計算
        # Calculating KL with Dirichlet prior and variational posterior distributions
        # Original paper:"Autoencodeing variational inference for topic model"-https://arxiv.org/pdf/1703.01488
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - K)        
        return BCE + KLD
# %%
model = Dir_VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# %%
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # data = transforms.Resize(120)(data)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, gauss_z, dir_z = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar, category)
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print(f"gause_z:{gauss_z[0]}") # Variables following a normal distribution after Laplace approximation
            #print(f"dir_z:{dir_z[0]},SUM:{torch.sum(dir_z[0])}") # Variables that follow a Dirichlet distribution. This is obtained by entering gauss_z into the softmax function
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
# %%
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, gauss_z, dir_z = model(data)
            loss = model.loss_function(recon_batch, data, mu, logvar, category)
            test_loss += loss.mean()
            test_loss.item()
            if i == 0:
                n = min(data.size(0), 18)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 
                           f'image/recon_{str(epoch)}_{gauss_z.shape[1]}cat.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
# %%
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, category).to(device)
        sample = model.decode(sample).cpu()

# %%
c = 0
encodings, labels = [], []
for imgs, label in train_loader:
    imgs = imgs.to(device)
    mu, logvar = model.encode(imgs)
    gauss_z = model.reparameterize(mu, logvar)
    recon_batch, mu, logvar, gauss_z, dir_z = model(imgs)

    encodings.append(gauss_z.cpu().detach().numpy())
    labels.append(label.cpu().detach().numpy())
    c +=1
    if c > 25:
        break
encodings = np.concatenate(encodings)
labels = np.concatenate(labels)
# %%
idx = 1
img0 = imgs[idx][0].detach().cpu().numpy()
label0 = label[idx].detach().cpu().numpy()
decoding0 = recon_batch[idx][0].detach().cpu().numpy()
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
def print_sizes(model, input_tensor):
    output = input_tensor
    for m in model.children():
        output = m(output)
        print(f'{m} \n {output.shape} \n')
    # return output
imgs, label = next(iter(train_loader))
imgs = imgs.to(device)
imgs = transforms.Resize(120)(imgs)
print_sizes(model.encoder, imgs)

# print_sizes(model.encoder, transforms.Resize(120)(imgs))
# %%
m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)
# %%
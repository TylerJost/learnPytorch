# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
train = datasets.MNIST("", train=True, download=True ,transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True ,transform = transforms.Compose([transforms.ToTensor()]))

# %%
# pytorch uses this dataloader

trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=100, shuffle=True)

# %%
# We can visualize this data
for data in trainset:
    print(data)
    break

# %%
# So each "data" is a batch
len(data[0])

# %%
# Tensor shapes for pytorch are odd
x, y = data[0][0], data[1][0]
print(x.shape)

# %%
# plt.imshow(x[0])
plt.imshow(x.view(28,28))

# %% [markdown]
# ## Building the Neural Network

# %%
import torch.nn as nn
import torch.nn.functional as F


# %%
class Net(nn.Module):
    def __init__(self):
        # Run initialization for nn.Module
        super().__init__()
        # 784 is the total number of pixels in the images (28x28), 64 is the output size
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        # Output 10 because there are 10 classes 0-9
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        Defines how the data will move forward through the network
        """
        
        # We pass through an activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # We wouldn't want relu/any activation function on the output
        x = self.fc4(x)
        
        # 
        return F.log_softmax(x, dim=1)
net = Net()
print(net)

# %%
X = torch.rand((28,28))
X = X.view(1, 28*28)


# %%
output = net(X)

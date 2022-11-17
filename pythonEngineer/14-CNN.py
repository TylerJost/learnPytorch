# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
# %matplotlib inline

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 8
batch_size = 4
learning_rate = 0.001

# PILImage images of range [0,1]
# Transform to tensors of normalized [-1, 1]
transform = transforms.Compose(
[transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = True)

classes = ('plate', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
# Make the conv net
"""
Start with convolution + RELU
Max Pooling (2x2)
Convolution + RELU
Max pooling
3 fully connected layers
Softmax/cross-entropy
"""

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(in_features = 16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features = 120, out_features=84)
        self.fc3 = nn.Linear(84, 10)
    def forward (self, x):
        # First convlutional and pooling layer
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # No activation function at end or softmax because it is in CrossEntropyLoss
        return x


# %%
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %%
print('Finished Training')
PATH = './data/cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

# %% [markdown]
# ## Explaining structure of CNN

# %%
dataiter = iter(train_loader)
images, labels = dataiter.next()

img = torchvision.utils.make_grid(images)/2+0.5
npimg = img.numpy()
plt.imshow(np.transpose(npimg,(1,2,0)))
plt.show()

# %%
conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
pool = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(6, 16, 5)

print(f'Initial shape: {images.shape}')
x = conv1(images)
print(f'After Conv1: {x.shape}')
x = pool(x)
print(f'After pooling: {x.shape}')
x = conv2(x)
print(f'After Conv2: {x.shape}')
x = pool(x)
print(f'After pooling: {x.shape}')
# Now we know why the input to the first fully connected layer is 16*5*5

# %% [markdown]
# Resulting image size is dependent on:
# $$
# (W-F + 2P)/S + 1
# $$
# Where:
#
# W = Input width
#
# F = Filter size
#
# P = Padding
#
# S = Stride
#
# %%
outputSizeCNN = lambda W, F, P, S: (W-F+2*P)/S + 1
outputSizePool = lambda W, F, S : (W-F)/S + 1
W = 32
F = 5 
P = 0
S = 1
W = outputSizeCNN(W, F, P, S)
W = outputSizePool(W, F=2, S=2)
# %% [markdown]
"""
# LeNet
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
from tqdm import tqdm
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters
num_epochs = 8
batch_size = 4
learning_rate = 0.001
# %%
transform = transforms.Compose(
[transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='../pythonEngineer/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='../pythonEngineer/data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = True)
# %%
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=5*16*5, out_features=120) # in - 400
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        sig = nn.Sigmoid()
        x = self.pool(sig(self.conv1(x)))
        x = self.pool(sig(self.conv2(x)))
        # Flatten
        x = x.view(-1, 5*16*5)
        x = sig(self.fc1(x))
        x = sig(self.fc2(x))
        x = self.fc3(x)
        return x
# %%
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
allLoss = []
for epoch in range(num_epochs):
    print(f'epoch: {epoch} ----')
    for i, (images, labels) in enumerate(tqdm(train_loader)):

        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        allLoss.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# %%
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# %%
plt.imshow(images[0][0].cpu())

# %%
_, b = torch.max(outputs[0])

# %%
dataiter = iter(train_loader)
images, labels = dataiter.next()

conv1 = nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=5, padding=2)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
fc1 = nn.Linear(in_features=5*16*5, out_features=120)
fc2 = nn.Linear(in_features=120, out_features=84)
fc3 = nn.Linear(in_features=84, out_features=10)

x = pool(conv1(images))

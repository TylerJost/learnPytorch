# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm 

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
# %% Load and reshape files
with h5py.File('../data/full_dataset_vectors.h5', "r") as hf:
    X_train = hf['X_train'][:]
    X_test = hf['X_test'][:]
    y_train = hf['y_train'][:]
    y_test = hf['y_test'][:]

# Each image should be 16x16x16
X_train = np.array([num.reshape([16,16,16]) for num in X_train])
X_test =  np.array([num.reshape([16,16,16]) for num in X_test])

# Visualize
# %% Visualize
idx = 102
num = X_train[idx]
x, y, z = num.nonzero()
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(x, y, z, c = z)
plt.title(y_train[idx])
plt.xlabel('x')
plt.ylabel('y')
# %% Build loader
class num3D(Dataset):
    def __init__(self, points, labels):
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).float()
        self.points = points
        self.labels = labels
    def __getitem__(self, idx):
        ptIdx = torch.from_numpy(np.expand_dims(self.points[idx], axis=0)).float()
        labelIdx = self.labels[idx]
        return [ptIdx, labelIdx]
    def __len__(self):
        return len(self.labels)
# %% Understanding inputs

train_dataset = num3D(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=4)
# %%
dataiter = iter(train_loader)
img, lbl = next(dataiter)
# %%
conv1 = nn.Conv3d(in_channels = 1, out_channels=32, kernel_size=(3,3,3))
# %%
class convNet3D(nn.Module):
    def __init__(self):
        super(convNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels = 1, out_channels=32, kernel_size=(3,3,3))
        self.conv2 = nn.Conv3d(in_channels = 32, out_channels=64, kernel_size=(3,3,3))
        self.pool = nn.MaxPool3d((2,2,2))
        self.conv3 = nn.Conv3d(in_channels = 64, out_channels=64, kernel_size=(3,3,3))
        self.conv4 = nn.Conv3d(in_channels = 64, out_channels=16, kernel_size=(3,3,3))
        self.flatten = torch.nn.Flatten()
        self.fc = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        bn = nn.BatchNorm3d(32, device='cuda')
        x = F.relu(bn(self.conv1(x)))

        bn = nn.BatchNorm3d(64, device='cuda')
        x = F.relu(bn(self.conv2(x)))

        x = F.relu(self.pool(self.conv3(x)))

        x = F.relu(bn(self.conv3(x)))
        
        bn = nn.BatchNorm3d(16, device='cuda')
        x = F.relu(bn(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
# %% Model parameters
num_epochs = 50
batch_size=64
# %% Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = convNet3D().to(device)
model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# %%
phase = 'train'
for epoch in range(num_epochs):
    n_correct = 0
    n_samples = 0
    for img, label in tqdm(train_loader):
        label = label.type(torch.LongTensor)
        img = img.to(device)
        label = label.to(device)
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
        
        n_samples += label.size(0)
        n_correct += (label == preds).sum().item()

        loss = criterion(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch accuracy: {n_correct/n_samples:0.2f}')
# %%

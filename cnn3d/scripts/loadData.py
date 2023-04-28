# %%
import numpy as np
import h5py
import matplotlib.pyplot as plt

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
        print(ptIdx)
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
conv1(img)

# %%


with h5py.File('../data/full_dataset_vectors.h5', "r") as hf:
    X_train = hf['X_train'][:]
    X_test = hf['X_test'][:]
    y_train = hf['y_train'][:]
    y_test = hf['y_test'][:]

def prepare_points(tensor):
    tensor = tensor.reshape((
            tensor.shape[0], 
            16, 
            16, 
            16
        ))
    return tensor

# apply threshold and reshaping given tensors
X_train = prepare_points(X_train)
X_test = prepare_points(X_test)

# %%
class dset(torch.utils.data.Dataset):
    def __init__(self, points, labels) -> None:
        super(Dataset, self).__init__()
        self.points = points
        self.labels = labels
        
    def __getitem__(self, index : int) -> torch.tensor:
        current_points, current_label = self.points[index], self.labels[index]
        current_points = torch.from_numpy(np.expand_dims(current_points, axis = 0))
        # current_label = torch.from_numpy(current_label)
        return [current_points, current_label]
    
    def __len__(self) -> int:
        return len(self.labels)
train_dataset = dset(X_train, y_train)
train_loader2 = DataLoader(train_dataset, batch_size=4)
dataiter = iter(train_loader2)
img2, lbl = next(dataiter)
# %%
# apply one-hot-encoding to given labels
encoder = OneHotEncoder(sparse = False)
y_train = encoder.fit_transform(y_train.reshape((y_train.shape[0], 1)))
y_test = encoder.fit_transform(y_test.reshape((y_test.shape[0], 1)))


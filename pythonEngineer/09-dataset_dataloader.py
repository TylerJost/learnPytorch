# %% [markdown]
"""
# Dataloaders
This is 
"""
# %%
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
# %%
class WineDataset(Dataset):
    """
    WineDataset inherits from pytorch's Dataset class
    This allows us to load the dataset on initialization, get an iterm, and return the length
    """
    def __init__(self):
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])

        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index] # Returns a tuple
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

first_data = dataset[0]

features, labels = first_data

print(features, labels)
# %%
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

dataiter = iter(dataloader)
data = dataiter.next()
features, labels = data
print(features, labels)

# %% training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = np.ceil(total_samples/4)

for epoch in range(num_epochs):
    # Iterate through the data loader for given number of epochs
    # The size of inputs is dependent on the batch size
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
        print(labels)
        break
# %%

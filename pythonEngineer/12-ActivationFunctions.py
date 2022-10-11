# %% [markdown]
"""
This is mostly an introduction to actually building the neural network (finally!)
"""
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
# %% Option 1 - Create the neural network modules and call them in the forward method
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLu()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# %% Option 2 - Use the activation functions directly in the forward pass
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


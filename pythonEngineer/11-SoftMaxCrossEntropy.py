# %% [markdown]
# %%
import torch
import torch.nn as nn
import numpy as np
# %% Softmax numpy
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

x = np.array([2, 1, 0.1])
outputs = softmax(x)
print('Softmax numpy:', outputs)
# %% Softmax pytorch
x = torch.tensor(x)
outputs = torch.softmax(x, -1)
print('Softmax pytorch: ', outputs)
# %% [markdown]
"""
Cross-Entropy is a commonly used loss function with probabilities
Labels must be one-hot encoded, and predicted must be probabilities
"""

# %%
def cross_entropy(y, yhat):
    loss = -np.sum(y*np.log(yhat))
    return loss

# y must be one hot encoded. 
# Class 0: [1, 0, 0]
# Class 1: [0, 1, 0]
# Class 2: [0, 0, 1]

y = np.array([1, 0, 0])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)

print(f'l1: {l1:0.3f}, l2: {l2:0.3f}')
# %% [markdown]
"""
Warnings:\
nn.CrossEntropyLoss == nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)\
Therefore, we don't use softmax in the last layer\


ALSO Y has class labels, not one-hot\

y_pred has raw scores (logits), no softmax!
"""
# %%
loss = nn.CrossEntropyLoss()
# Y is of size nsamplesxnclasses = 1x3
y = torch.tensor([2, 0, 1]) # Class 0
y_pred_good = torch.tensor([[2.0, 1.0, 9.1], [2.0, 1.0, 0.1], [2.0, 10.0, 0.1]])
y_pred_bad = torch.tensor([[0.5, 2, 0.3], [0.5, 2, 0.3], [0.5, 2, 0.3]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(f'l1: {l1.item():0.3f}, l2: {l2.item():0.3f}')
# %%
_, prediction1 = torch.max(y_pred_good, 1)
_, prediction2 = torch.max(y_pred_bad, 1)

print(prediction1, prediction2)
# %% Building a multiclass neural net

class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at end
        return out

model = NeuralNet2(input_size = 28*28, hidden_size = 5, num_classes = 3)
criterion = nn.CrossEntropyLoss() # Applies softmax


# %% Binary classification example
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet2(input_size = 28*28, hidden_size = 5, num_classes = 3)
criterion = nn.BCELoss() # Applies softmax
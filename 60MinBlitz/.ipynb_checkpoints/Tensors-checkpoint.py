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

# %% [markdown]
# # Tensors
# Basically a numpy array that is specialized for GPUs. 

# %%
import torch, torchvision
import numpy as np

# %% [markdown]
# ## Making Tensors

# %%
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# %%
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# %%
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %% [markdown]
# ## Tensor Attributes

# %%
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %%
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")

# %% [markdown]
# Tensors are pretty similar to numpy arrays, so it really shouldn't be too big of a deal. 

# %% [markdown]
# # Using `TORCH.AUTOGRAD`
# This is the automatic differentiation used for backprop in PyTorch

# %%
# Load pretrained model
model = torchvision.models.resnet18(pretrained=True)
# Simulate an image with 3 channels, and a height/width of 64
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# %%
# Forward pass
prediction = model(data)

# %%
# Basic loss
loss = (prediction - labels).sum()
# Backward pass
loss.backward()
print(loss)

# %%
# Load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# Initiate gradient descent
optim.step()

# %% [markdown]
# ### Autograd Details - Differentiation

# %%
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# %% [markdown]
# Create a new tensor `Q` from `a` and `b`
# $$
# Q = 3a^3 - b^2
# $$ 

# %%
Q = 3*a**3 - b**2

# %% [markdown]
# Find partial derivatives:
# $$
# \frac{\partial{Q}}{\partial{a}} = 9a^2
# \\
# \frac{\partial{Q}}{\partial{b}} = -2b
# $$

# %% [markdown]
# Calling `.backward()` on `Q` will calculate the gradients and store them in `.grad`
#
# From tutorial:
# "We need to explicitly pass a gradient argument in Q `.backward()` because it is a vector. `gradient` is a tensor of the same shape as Q, and it represents the gradient of Q w.r.t. itself, i.e.
#
# $$
# \frac{\partial{Q}}{\partial{Q}} = 1
# $$
#
#

# %%
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# %%
print(a.grad)
print(b.grad)

# %% [markdown]
# Makes sense! Eg:
# $$
# \frac{\partial{Q}}{\partial{a}} = 9a^2 = 9*4^2 = 36
# $$

# %% [markdown]
# We don't have to use autograd on everything. For example, we could finetune a pretrained network by "freezing" most of the model. Let's load and freeze a model. 

# %%
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# %% [markdown]
# For this, the classifier is the last layer called `model.fc`. We can replace it with a new layer which is unfrozen by default. 

# %%
model.fc = nn.Linear(512, 10)

# %%
# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

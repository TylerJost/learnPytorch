# %% [markdown]
"""
This notebook will demonstrate the training loop in numpy, then update the gradients using pytorch.
In the next notebook (6) 
"""
# %%
import numpy as np
import torch
# %% Numpy only
# Linear regression:
# f = w*x+b

# For this problem, f = 2*x

x = np.array([1,2,3,4])
y = 2*x

# Initial weight 
w = 0  

# Model prediction
# Forward pass is a very pytorch way to do this
def forward(x):
    return w*x
# Loss - MSE here
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()
# Gradient - Wrt our parameters
# MSE = 1/N * (w*x - y)^2
# dJ/dw = 1/N*2x*(w*x-y)
def gradient(x, y, y_pred):
    N = len(x)
    # return sum(1/N*2*x*(y_pred-y))
    return (2*x * (y_pred - y)).mean()

print(f'Prediction before training: f(5) = {forward(5):0.3f}')

# Training
lr = 0.01
n_iters = 20

for epoch in range(n_iters):
    # Prediction
    y_pred = forward(x)

    l = loss(y, y_pred)

    dJdw = gradient(x, y, y_pred)

    # update weights
    w -= lr*dJdw

    if epoch % 1 == 0:
        print(f'epoch: {epoch} \t weight={w:0.2f} \t loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):0.3f}')

# %% With pytorch instead for the gradient
x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = 2*x

# Initial weight 
w = torch.tensor(0, dtype=torch.float32, requires_grad=True)
# Model prediction
# Forward pass is a very pytorch way to do this
def forward(x):
    return w*x

# Loss - MSE here
def loss(y, y_pred):
    return ((y_pred-y)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):0.3f}')
# %%
# Training
lr = 0.01
n_iters = 200

for epoch in range(n_iters):
    # Prediction
    y_pred = forward(x)

    l = loss(y, y_pred)

    # Backward pass
    l.backward() #dl/dw

    # update weights
    with torch.no_grad():
        w -= lr*w.grad

    # Must zero the gradients because l.backward() will accumulate gradients in w.grad()
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch: {epoch} \t weight={w:0.2f} \t loss = {l:.8f}')

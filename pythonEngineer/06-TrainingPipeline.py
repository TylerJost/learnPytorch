# %% [markdown]
"""
Steps in the training pipeline:
1. Design model (input, output, forward pass)
2. Loss and optimizer
3. Training Loop:
  * Forward pass: Compute predictions
  * Backward pass: Get the gradients
  * Update weights
  * Repeat!

In this script I'll further replace loss/optimization with pytorch
"""
# %%
import numpy as np
import torch
import torch.nn as nn

# %%
# For pytorch 
x = torch.tensor([1,2,3,4], dtype=torch.float32).view(4,1)
y = torch.tensor(2*x).view(4,1)

xTest = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = x.shape
# This is built in
model = nn.Linear(n_features, n_features)

# Practice writing our own model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__() #?
        # Define layers
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(n_features, n_features)

print(f'Prediction before training: f({xTest[0]}) = {model(xTest).item():0.3f}')

# Training
lr = 0.01
n_iters = 50

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(n_iters):
    # Prediction
    y_pred = model(x)

    l = loss(y, y_pred)

    # Backward pass
    l.backward() #dl/dw

    # update weights
    optimizer.step()

    # Must zero the gradients because l.backward() will accumulate gradients in w.grad()
    optimizer.zero_grad()

    if epoch % 1 == 0:
        [w, b] = model.parameters()
        print(f'epoch: {epoch} \t weight={w[0][0].item():0.2f} \t loss = {l:.8f}')

print(f'Prediction after training: f({xTest[0]}) = {model(xTest).item():0.3f}')


# %%

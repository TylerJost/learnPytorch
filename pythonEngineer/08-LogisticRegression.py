# %% [markdown]
"""
Practice with a new type of function/loss function
Recall:

Steps in the training pipeline:
1. Design model (input, output, forward pass)
2. Loss and optimizer
3. Training Loop:
  * Forward pass: Compute predictions
  * Backward pass: Get the gradients
  * Update weights
  * Repeat!
"""
# %%
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# %%
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1) # Make into a column vector
y_test = y_test.view(y_test.shape[0], 1) # Make into a column vector

# Model
# f = wx+b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super().__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)
# Loss/optimizer
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# %%
# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(X_train)

    l = loss(y_pred, y_train)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'Loss: {l:.8f}')

with torch.no_grad():
    y_pred = model(X_test)
    accuracy = y_pred.round().eq(y_test).sum()/y_test.shape[0]
    print(f'Accuracy {accuracy.item()*100:.4}%')
# %%

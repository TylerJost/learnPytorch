# %%
import torch
import torch.nn as nn
# %% [markdown]
# ## State Dict


# %%
"""
torch.save(model.state_dict(), PATH)

# Model must be created again with parameters

"""


# %%
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

# %%
PATH = './models/model.pth'
torch.save(model.state_dict(), PATH)

# %%
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(PATH))

# %%
for param in model.parameters():
    print(param)
print(20*'-')
for param in loaded_model.parameters():
    print(param)

# %%
print(model.state_dict())

# %%
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())

# %% [markdown]
# ### Saving a model checkpoint with optimizer and epoch num:

# %%
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

torch.save(checkpoint, './models/checkpoint.pth')

# %%
loaded_checkpoint = torch.load("./models/checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0) # Note this will be overwritten

model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])

print(optimizer.state_dict()) # Notice learning rate is now 0.01
print(model.state_dict())


# %% [markdown]
# ### Loading and saving a model with gpu

# %%
device = torch.device("cuda")
model.to(device)
torch.save()

# %%
# !squeue -u tjost

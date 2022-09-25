import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline
REBUILD_DATA=False


# +
class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    
    catcount = 0
    dogcount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    # This conversion is not completely necessary for CNN
                    # However, color is not really relevant to if something is a cat/dog
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Resize
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    # We want the image, as well as a ones-hot array
                    # [cat, dog]
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                # Apparently there are some weird images that cannot be loaded (?)
                except Exception as e:
                    print(path)
                    os.remove(path)
#                     print('There is an error!:')
#                     print(str(e))
                    pass
        np.random.seed(1234)
        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print(f'Cats: {self.catcount}   Dogs: {self.dogcount}')

if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()
# -

training_data = np.load('training_data.npy', allow_pickle=True)
plt.imshow(training_data[1][0], cmap='gray')
plt.show()

# # Build the Model

import torch
import torch.nn as nn
import torch.nn.functional as F

training_data[1][0].shape


# +
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Basically do the same thing as before
        # Input, Output, Convolutional size
        self.conv1 = nn.Conv2d(1, 32, 5) #inputs 1, outputs 32 using a 5x5
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512) # Flattening
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 classes out
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
#         print(self._to_linear)
#         print(x.flatten().shape[0])
        
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # Recall that .view == reshape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()

# +
import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0 # Scales values
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

# +
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[val_size:]
test_y = y[val_size:]

# -

batch_size = 100
epochs = 1
for epoch in range(epochs):
    for i in tqdm(range(0, len(train_X), batch_size)):
        batch_X = train_X[i:i+batch_size].view(-1,1,50,50)
        batch_y = train_y[i:i+batch_size]
        
        net.zero_grad()
#         optim.zero_grad() # The optimizer controls all paramters for this model, so this is equivalent
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

correct/total

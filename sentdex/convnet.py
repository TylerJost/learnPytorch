##
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
# %matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA=False

##
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on GPU")

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
##

training_data = np.load('training_data.npy', allow_pickle=True)
# plt.imshow(training_data[1][0], cmap='gray')
# plt.show()

# # Build the Model

##
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
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # Recall that .view == reshape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

net = Net()

##
X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0 # Scales values
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

##
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[val_size:]
test_y = y[val_size:]
##

net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

##
MODEL_NAME = f"model-{int(time.time())}"

def fwd_pass(X, y, train=False):
    """
    Generic function for:
    taking in data and training, 
    running through the network, 
    then calculating metrics
    """
    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50).to(device), y.to(device))
    return val_acc, val_loss

def train(net):
    BATCH_SIZE = 100
    EPOCHS = 3

    with open(f"./logs/{MODEL_NAME}.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
                #print(f"{i}:{i+BATCH_SIZE}")
                # Generally not all the data can fit on the GPU
                # So we send data to a batch

                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)

                net.zero_grad()

                # Send through network
                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 10 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{time.time():0.4f},{acc:0.4f},{loss:0.4f},{val_acc:0.4f},{val_loss:0.4f}\n")
            print(f"Epoch: {epoch} \n Loss: {loss:0.2f} In-sample acc: {acc:0.2f}")
train(net)
##

# ! squeue -u tjost

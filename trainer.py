import torch
import numpy as np
from randlanet import RandLANet
import os
from utils import LidarDataset,targets_dict
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


lidar_path = "/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/BV1_Circuit_Beaver_Creek_1_READY/cloud328819b9aee51228.las"
dataset = LidarDataset(lidar_path)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
dataset.to_device(device)
dataloader = DataLoader(dataset, batch_size=1000000)

#make map of classification to 0-num_classes
fn_dict = targets_dict(dataset.targets) #dictionary
#rewrite this so it does not reference above variable... might have to get rid of apply_ function in training loop
def targets_map(x):
    return fn_dict[x] 

#model parameters
d_in = dataset.features.shape[1]
num_classes = np.unique(dataset.targets).shape[0]

#model training config
model = RandLANet(d_in, num_classes, 16, 4, device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)
epochs = 40

train_loss_list = []
val_loss_list = []
print("starting training loop")
for epoch in range(epochs):
    print(epoch)
    for x,y in dataloader:
        x = x.reshape(1,-1,7)
        y = y.apply_(targets_map).reshape(-1)

        z = model(x)
        yhat = z.squeeze(0).T
        loss = criterion(yhat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(loss)
    train_loss_list.append(loss)

fig, ax = plt.subplots()
ax.plot(train_loss_list)
plt.show()
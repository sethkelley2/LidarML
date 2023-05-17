import torch
import numpy as np
from randlanet import RandLANet
import os
from utils import LidarDataset,targets_dict
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

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

#validation dataset
val_path = "/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/BV1_Circuit_Beaver_Creek_1_READY/cloud328819ad3d77f188.las"
val_dataset = LidarDataset(val_path)
val_dataset.to_device(device)
val_dataloader = DataLoader(val_dataset,batch_size=1000000)

#loss list
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
    train_loss_list.append(loss.item())

    with torch.no_grad():
        for x,y in val_dataloader:
            x = x.reshape(1,-1,7)
            y = y.apply_(targets_map).reshape(-1)

            z = model(x)
            yhat = z.squeeze(0).T
            loss = criterion(yhat,y)

    val_loss_list.append(loss.item())

    print(train_loss_list,val_loss_list)

with open('train_loss_list.pkl', 'wb') as file1:
    pickle.dump(train_loss_list, file1)

# Pickle and save list2
with open('val_loss_list.pkl', 'wb') as file2:
    pickle.dump(val_loss_list, file2)

fig, ax = plt.subplots()
ax.plot(train_loss_list, label='Train Loss')
ax.plot(val_loss_list, label='Validation Loss')
ax.legend()  # Add a legend to differentiate the lines
plt.show()
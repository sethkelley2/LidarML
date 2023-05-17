import torch
import numpy as np
from randlanet import RandLANet
import os
from utils import LidarDataset
from torch.utils.data import DataLoader
import torch.nn as nn

lidar_path = "/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/BV1_Circuit_Beaver_Creek_1_READY/cloud328819b9aee51228.las"
dataset = LidarDataset(lidar_path)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
dataset.to_device(device)
dataloader = DataLoader(dataset, batch_size=100000)

#model parameters
d_in = dataset.features.shape[1]
num_classes = np.unique(dataset.targets).shape[0] + 1

#model training config
model = RandLANet(d_in, num_classes, 16, 4, device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)
epochs = 40

for epoch in range(epochs):
    for x,y in dataloader:
        x = x.reshape(1,-1,7)

        yhat = model(x)
        loss = criterion(yhat,)



print("evaluating")
# model(features)
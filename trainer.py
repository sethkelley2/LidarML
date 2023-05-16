import torch
import numpy as np
from randlanet import RandLANet
import os
from utils import las_to_npy

lidar_path = "/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/BV1_Circuit_Beaver_Creek_1_READY/cloud328819c226c3eab4.las"

features,classification = las_to_npy(lidar_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

d_in = features.shape[1]
num_classes = np.unique(classification).shape[0]

features = torch.tensor(features,dtype=torch.float32).reshape(1,-1,7)
features = features.to(device)

model = RandLANet(d_in, num_classes, 16, 4, device)

print("evaluating")
model(features)
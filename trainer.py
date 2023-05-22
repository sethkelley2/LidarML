import torch
import numpy as np
from randlanet import RandLANet
import os
from utils import LidarDataset,targets_dict,calculate_accuracy,targets_map,confusion
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from collections import Counter

lidar_path = "/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520c1fa4732de01.las"
dataset = LidarDataset(lidar_path)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")
dataset.to_device(device)
dataloader = DataLoader(dataset, batch_size=90000)

#model parameters
d_in = dataset.features.shape[1]
num_classes = np.unique(dataset.targets).shape[0]

down_sampling = 2
num_neighbors = 50

#model training config
model = RandLANet(d_in, num_classes, num_neighbors, down_sampling, device)
weights = [1,8,2,2,2]
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)
epochs = 3

#validation dataset
val_path = "/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520c1fb86512e5b.las"
val_dataset = LidarDataset(val_path)
val_dataset.to_device(device)
val_dataloader = DataLoader(val_dataset,batch_size=90000)

#loss list
train_loss_list = []
val_loss_list = []
print("starting training loop")
for epoch in range(epochs):
    print(epoch)
    for x,y in dataloader:
        x = x.reshape(1,-1,7).to(device)
        y = y.apply_(targets_map).reshape(-1).to(device)

        z = model(x)
        yhat = z.squeeze(0).T
        loss = criterion(yhat,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        confusion(np.array(yhat.argmax(dim=1).reshape(-1).detach().cpu()),np.array(y.reshape(-1).detach().cpu()))

        count = Counter(y.tolist())
        print(f"the baseline accuracy is {count.most_common(1)[0][1]/sum(count.values())}")

        acc = calculate_accuracy(yhat,y)
        print(f"the model accuracy in training is {acc}")

    print(loss)
    train_loss_list.append(loss.item())

    with torch.no_grad():
        for x,y in val_dataloader:
            x = x.reshape(1,-1,7).to(device)
            y = y.apply_(targets_map).reshape(-1).to(device)

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
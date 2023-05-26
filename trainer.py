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
from sklearn.preprocessing import StandardScaler,MinMaxScaler

dataset1 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520c1fa4732de01.las")
# dataset2 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520c10c3326d1e7.las")
# dataset3 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520ce2b194e0e4c.las")
# dataset4 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520ce5a6804b135.las")
# dataset5 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520ce9d3d522890.las")
#dataset6 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520cff643ba896e.las")
# dataset7 = LidarDataset("/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6537e26edc90bd18.las")

data = [dataset1]#,dataset4,dataset5,dataset6,dataset7]
device = torch.device("cpu")

#model parameters
d_in = dataset1.features.shape[1]
num_classes = np.unique(dataset1.targets).shape[0]

down_sampling = 2
num_neighbors = 16

#model training config
model = RandLANet(d_in, num_classes, num_neighbors, down_sampling, device)

weights = [1,1,8,8,8]
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)#,label_smoothing=0.3)
# criterion = nn.NLLLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.005)
epochs = 10

#validation dataset
val_path = "/media/lidarml/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/Toledo_Poles_Circuit_Yaquina_Bay_Road_1_READY/cloud6520c1fb86512e5b.las"
val_dataset = LidarDataset(val_path)
val_dataset.to_device(device)
val_dataloader = DataLoader(val_dataset,batch_size=60000)

#store predictions in rolling fashion
stacked_preds = []

#loss list
train_loss_list = []
val_loss_list = []
print("starting training loop")
for epoch in range(epochs):
    print(epoch)
    for set in data:

        set.to_device(device)
        dataloader = DataLoader(set, batch_size=60000)

        for x,y in dataloader:

            scaler = MinMaxScaler()
            scaler.fit(x)
            x = scaler.transform(x)
            x = torch.tensor(x,dtype=torch.float32)

            x = x.reshape(1,-1,11).to(device)
            y = y.apply_(targets_map).reshape(-1).to(device)#.apply_(label_smoothing)

            z = model(x)
            yhat = z.squeeze(0).T
            loss = criterion(yhat,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            #rolling stack of predictions
            if len(stacked_preds) >= 30:
                stacked_preds.insert(0,(scaler.inverse_transform(x.reshape((-1,11))),yhat))
                stacked_preds.pop()
            else:
                stacked_preds.insert(0,(x,yhat))


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
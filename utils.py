import numpy as np
import laspy
import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from typing import Tuple,List,Dict
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors

class LidarDataset(Dataset):
    ''' Dataset class for creating RandLANet compatible dataset.
        This class takes in a path to a classified las file and return
        a dataloading '''
    def __init__(self,las_path):
        super().__init__()
        self.path = las_path
        self.features, self.targets = self.las_to_npy(las_path)

        #normalize features to 0 mean with unit variance
        # self.scaler = MinMaxScaler()
        # self.scaler.fit(self.features)
        # self.features = self.scaler.transform(self.features)

        self.to_tensor()
    
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx,:], self.targets[idx]

    def las_to_npy(self, las_file):
        # Read the LAS file
        inFile = laspy.read(las_file)

        # Extract point cloud data (x, y, z)
        x = inFile.x
        y = inFile.y
        z = inFile.z
        red = inFile.red
        green = inFile.green
        blue = inFile.blue
        intensity = inFile.intensity
        angle = inFile.scan_angle_rank
        num_returns = inFile.number_of_returns
        direction = inFile.scan_direction_flag
        edgeline = inFile.edge_of_flight_line

        data = np.column_stack((x, y, z, intensity, red, green, blue,angle,num_returns,direction,edgeline))
        # data = np.column_stack((z,red, green, blue, intensity))


        # Assuming 'classification' is available in the LAS file
        classification = inFile.classification
        classification = np.array(classification).reshape(-1,1)

        indices = self.get_sorted_indices(data)

        data = data[indices]
        classification = classification[indices]

        return data,classification

    def get_sorted_indices(self,data):
        if (data[:,0].max()-data[:,0].min()) > (data[:,1].max() - data[:,1].min()):
            indices = np.lexsort((data[:,1], data[:,0]))
        else:
            indices = np.lexsort((data[:,0], data[:,1]))
        return indices

    def to_tensor(self):
        self.features = torch.tensor(self.features,dtype=torch.float32)
        self.targets = torch.tensor(self.targets,dtype=torch.int64)

    def to_device(self,device):
        self.features.to(device)
        self.targets.to(device)

def targets_dict(targets):
    mapper = dict()
    for i,elem in enumerate(np.unique(targets)):
        mapper[elem] = i
    return mapper

def targets_map(x):
    fn_dict = {5:0,2:0,14:2,15:3,7:4,13:5}
    return fn_dict[x] 

def confusion(yhat,y):
    cf_matrix = confusion_matrix(y,yhat)
    class_names = ('5','2','14','15','7','13')
    # Create pandas dataframe
    try:
        dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
        print(dataframe)
    except:
        print(cf_matrix)

def visualize_preds(data_list: List[Tuple]):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Colors to use for the classes
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']

    # Prepare empty lists to store points and classes
    all_points = []
    all_classes = []

    # Iterate through each tuple in the list
    for item in data_list:
        data,labels = item
        labels = labels.cpu().detach()
        # Get the points and corresponding classes
        points = data[:,0:3]
        classes = torch.argmax(labels, dim=1).detach().numpy()

        # Append points and classes to the respective lists
        all_points.append(points)
        all_classes.append(classes)

    # Concatenate all points and classes
    all_points = np.concatenate(all_points, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)

    # Assigning colors to classes
    color_map = [colors[c%len(colors)] for c in all_classes]

    # Plotting the data
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c=color_map, marker='o',s=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def calculate_accuracy(preds,actuals):
    preds = preds.argmax(dim=1)
    correct = 0
    total = preds.shape[0]
    for i in range(preds.shape[0]):
        if preds[i].item() == actuals[i].item():
            correct += 1
    return(np.round(correct/total,4))

def remove_outliers(data_list):
    all_points = []

    # Iterate through each tuple in the list
    for item in data_list:
        data,labels = item
        labels = labels.cpu().detach()
        # Get the points and corresponding classes
        points = data[:,0:3]
        labels = np.int_(torch.argmax(labels, dim=1).detach().numpy().reshape((-1,1)))
        points = np.hstack((points,labels))

        # Append points and classes to the respective lists
        all_points.append(points)

    # Concatenate all points and classes
    points = np.concatenate(all_points, axis=0)

    # We separate the coordinates and the classes for convenience
    coordinates = points[:, :3]
    classes = points[:, 3]

    # Create a nearest neighbors model (change n_neighbors to your needs)
    nn_model = NearestNeighbors(n_neighbors=16).fit(coordinates)

    # For each point, find its neighbors
    distances, indices = nn_model.kneighbors(coordinates)

    # Now we check for each point if it's class is the same as the majority of its neighbors
    # If not, we'll mark it for removal
    outliers = []
    for idx, neighbors in enumerate(indices):
        neighbor_classes = classes[neighbors]
        majority_class = max(set(neighbor_classes), key=list(neighbor_classes).count)
        if classes[idx] != majority_class:
            outliers.append(idx)

    # Now we remove the outliers from the original dataset
    filtered_points = np.delete(points, outliers, axis=0)
    return filtered_points

def dev_load():
    path = os.path.join(os.getcwd(),"stacked_preds.pkl")
    with open(path,'rb') as file:
        return (pickle.load(file))

def dev_plot(all_points):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Colors to use for the classes
    colors = ['g', 'b', 'r', 'c', 'm', 'y', 'k']

    all_classes = list(map(int,all_points[:,3].tolist()))

    # Assigning colors to classes
    color_map = [colors[c%len(colors)] for c in all_classes]

    # Plotting the data
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c=color_map, marker='o',s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":
    dataset = dev_load()
    classified = remove_outliers(dataset)
    dev_plot(classified)
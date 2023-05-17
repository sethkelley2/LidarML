import numpy as np
import laspy
import os
from torch.utils.data import Dataset
import torch


class LidarDataset(Dataset):
    ''' Dataset class for creating RandLANet compatible dataset.
        This class takes in a path to a classified las file and return
        a dataloading '''
    def __init__(self,las_path):
        super().__init__()
        self.path = las_path
        self.features, self.targets = self.las_to_npy(las_path)
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
        data = np.column_stack((x, y, z, intensity, red, green, blue))

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
        self.targets = torch.tensor(self.targets,dtype=torch.int16)

    def to_device(self,device):
        self.features.to(device)




if __name__ == "__main__":
    dataset = LidarDataset("/media/aerotractai/Archive100A/Wiggins_20230509/Aerotract (shared)/LiDARClassified/BV1_Circuit_Beaver_Creek_1_READY/cloud328819ad3d77f188.las")
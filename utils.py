import numpy as np
import laspy
import os

def las_to_npy(las_file):
    # Read the LAS file
    inFile = laspy.read(las_file)
    # Extract point cloud data (x, y, z)
    print("extract point cloud data")
    x = inFile.x
    y = inFile.y
    z = inFile.z

    red = inFile.red
    green = inFile.green
    blue = inFile.blue

    intensity = inFile.intensity

    # Assuming 'classification' is available in the LAS file
    print("extract classfication")
    classification = inFile.classification
    classification = np.array(classification).reshape(-1,1)

    data = np.column_stack((x, y, z, intensity, red, green, blue))

    return data,classification


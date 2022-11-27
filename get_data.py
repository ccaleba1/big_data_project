import os
import random
import torch
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data = []
for filename in os.listdir("data/1301/rgb"):

    file = "data/1301/rgb/" + str(filename)
    img = Image.open(file)
    data_array = np.array(img)
    data.append(data_array)

data = np.array(data)


with open('data_array.pkl', 'wb') as f:
    pickle.dump(data, f)

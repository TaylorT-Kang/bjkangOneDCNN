import torch.nn as nn
import numpy as np
import torch
import oneDCNN
import load_mat_data
from scipy import io
import hyperNomalize
import matplotlib.pyplot as plt

PATH = './bjkangNet.pth'
batch_size = 1
test_sample = 0.3
hyperCube, hyperGt, input_channels, n_classes, _, _ = load_mat_data.load_mat('./Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat',batch_size, test_sample)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

model = oneDCNN.bjkangNet(input_channels,n_classes)
model.load_state_dict(torch.load(PATH))
print(model)

HSI_data = np.array(hyperCube, dtype=np.float)
h, w, _ = HSI_data.shape
predic_map_include_zeroLabel = np.zeros((h,w))
predic_map = np.zeros((h,w))

with torch.no_grad():
    for i in range(h):
        for j in range(w):
            spectral = np.array(HSI_data[i,j,:])
            spectral = np.expand_dims(spectral,axis=0)
            spectral = np.expand_dims(spectral,axis=0)
            spectral = torch.from_numpy(spectral).float()
            output = model(spectral)
            a, b = torch.max(output,1)
            predic_map[i,j] = b
            if a >= 30:
                b = 0
            predic_map_include_zeroLabel[i, j] = b   
         
            if hyperGt[i,j] == 0:
                predic_map[i, j] = 0


fig = plt.figure(1,figsize=(10,5))
ax = fig.add_subplot(1,3,1)
im = ax.imshow(hyperGt,aspect='auto',cmap='jet')
fig.colorbar(im)
ax.set_xlabel('Ground Truth')
ax = fig.add_subplot(1,3,2)
im = ax.imshow(predic_map_include_zeroLabel, aspect='auto',cmap='jet')
fig.colorbar(im)
ax.set_xlabel('predict with zero label')
ax = fig.add_subplot(1,3,3)
im = ax.imshow(predic_map, aspect='auto',cmap='jet')
fig.colorbar(im)
ax.set_xlabel('predict without zero label')
plt.show()

fig.savefig('dection.png',dpi=300)
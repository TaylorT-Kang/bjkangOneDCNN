import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gradcam
import oneDCNN
import load_mat_data

PATH = './bjkangNet.pth'
batch_size = 100
train_sample = 0.7
input_channels, n_classes, train_loader, test_loader = load_mat_data.load_mat('./Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat',batch_size, train_sample)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

model = oneDCNN.bjkangNet(input_channels,n_classes)
model.load_state_dict(torch.load(PATH))


grad_cam = gradcam.GradCam(model=model, feature_module=model.conv5, \
                    target_layer_names=["conv5"], use_cuda=DEVICE)


## test git hub
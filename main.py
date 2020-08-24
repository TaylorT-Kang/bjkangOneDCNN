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
batch_size = 1
test_sample = 0.3
input_channels, n_classes, train_loader, test_loader = load_mat_data.load_mat('./Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat',batch_size, test_sample)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

model = oneDCNN.bjkangNet(input_channels,n_classes)
model.load_state_dict(torch.load(PATH))


grad_cam = gradcam.GradCam(model=model, feature_module=model, \
                    target_layer_names=["conv5"], use_cuda=DEVICE)
gb_model = gradcam.GuidedBackpropReLUModel(model=model, use_cuda=DEVICE)
test_spectrals, labels = iter(test_loader).next()
input_ = test_spectrals.requires_grad_(True)
# mask = grad_cam(input_,None)
gb_list = {}


dataiter = iter(test_loader)
for _, data in enumerate(dataiter):
    spectral, label = data[0].to(DEVICE), data[1].to(DEVICE)
    spectral = spectral.requires_grad_(True)
    gb = gb_model(spectral,index=label)
    key = label.cpu().numpy()
    key = key.tolist()[0]
    if key not in gb_list.keys():
        gb_list[key] = []
    gb_list[key].append(gb)
    breaking_point = 0


plt.figure(1)
plt.imshow(gb, aspect='auto')
# plt.colorbar()
plt.show()
plt.close()


# target_index = torch.unsqueeze(lable,0)
# target_index = target_index.to(DEVICE)
# input_ = input_.requires_grad_(True)
# print(model._modules.items())
# print(input_)
# gb_model = gradcam.GuidedBackpropReLUModel(model=model, use_cuda=DEVICE)
# gb = gb_model(input_, index=target_index)
# print(gb)
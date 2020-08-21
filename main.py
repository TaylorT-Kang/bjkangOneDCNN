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
test_sample = 0.7
input_channels, n_classes, train_loader, test_loader = load_mat_data.load_mat('./Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat',batch_size, test_sample)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

model = oneDCNN.bjkangNet(input_channels,n_classes)
model.load_state_dict(torch.load(PATH))


grad_cam = gradcam.GradCam(model=model, feature_module=model.conv5, \
                    target_layer_names=["conv5"], use_cuda=DEVICE)


test_spectrals, labels = iter(test_loader).next()
label = labels[0]
input_ = test_spectrals.requires_grad_(True)

# intput_np = input_.to("cpu").numpy()
gb_model = gradcam.GuidedBackpropReLUModel(model=model, use_cuda=DEVICE)
target_index = label
print("target : " , label)
# target_index = torch.unsqueeze(label,0)
target_index = target_index.to(DEVICE)
gb = gb_model(input_, index=target_index)
# print(gb)
plt.figure(1, figsize=(5,5))
plt.imshow(gb)
plt.colorbar()
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
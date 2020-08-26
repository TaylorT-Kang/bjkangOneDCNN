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

grad_cam = gradcam.GradCam(model=model, feature_module=model, target_layer_names=["conv5"], use_cuda=DEVICE)
gb_model = gradcam.GuidedBackpropReLUModel(model=model, use_cuda=DEVICE)
# test_spectrals, labels = iter(test_loader).next()
# input_ = test_spectrals.requires_grad_(True)
# mask = grad_cam(input_,None)
# gb = gb_model(test_spectrals,index=None)

gb_list = {}
grad_cam_list = {}

dataiter = iter(test_loader)
for _, data in enumerate(dataiter):
    spectral, label = data[0].to(DEVICE), data[1].to(DEVICE)
    spectral = spectral.requires_grad_(True)
    gb = gb_model(spectral,index=None)
    mask = grad_cam(spectral,index=None)
    key = label.cpu().numpy()
    key = key.tolist()[0]
    if key not in gb_list.keys():
        gb_list[key] = []
    if key not in grad_cam_list.keys():
        grad_cam_list[key]=[]
    gb_list[key].append(gb)
    grad_cam_list[key].append(mask)

gb_avg = {}

for key, val in gb_list.items():
    val_array = np.array(val)
    gb_avg[key] = np.expand_dims(sum(val_array[:,0]) / len(val),axis=0)

grad_cam_avg = {}
for key, val in gb_list.items():
    val_array = np.array(val)
    grad_cam_avg[key] = np.expand_dims(sum(val_array[:,0]) / len(val),axis=0)


guided_grad_cam = {}
for key in gb_avg.keys():
    guided_grad_cam[key] = gb_avg[key] * grad_cam_avg[key]

number_of_key = len(guided_grad_cam)
fig = plt.figure(1)
cols = 1
rows = number_of_key
i = 1
for key, val in guided_grad_cam.items():
    ax = fig.add_subplot(rows,cols,i)
    ax.imshow(val,aspect='auto')
    ax.set_ylabel(key)
    i += 1

plt.show()
# # plt.figure(1)
# # plt.title('backpro')
# # plt.imshow(gb_avg[1], aspect='auto')
# # # plt.colorbar()
# plt.figure(2)
# plt.title('gradcam')
# plt.imshow(grad_cam_avg[1],aspect='auto')

# plt.show()
# plt.close()
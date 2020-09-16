import torch.nn as nn
import numpy as np
import torch
import oneDCNN
import load_mat_data
from scipy import io
import hyperNomalize
import matplotlib.pyplot as plt

if __name__=='__main__':

    PATH = './bjkangNet_add.pth' #2020_ 9_ 11_ 16_ 49_ 13
    PATH = './Result/2020_ 9_ 11_ 16_ 49_ 13/bjkangNet_add.pth'
    batch_size = 1
    test_sample = 0.3
    hyperCube, hyperGt, input_channels, n_classes, _, test_loader = load_mat_data.load_mat('./Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat',batch_size, test_sample)


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = oneDCNN.bjkangNet(input_channels,n_classes)
    model.load_state_dict(torch.load(PATH))
    print(model)

    model.to(DEVICE)
    test_loss, test_accuracy = oneDCNN.evaluate(model, test_loader,0,DEVICE)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

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
                spectral = spectral.to(DEVICE)
                output = model(spectral)
                a, b = torch.max(output,1)
                predic_map[i,j] = b
                # if a >= 30:
                #     b = 0
                predic_map_include_zeroLabel[i, j] = b   
            
                if hyperGt[i,j] == 0:
                    predic_map[i, j] = 0


    fig = plt.figure(1,figsize=(13,4))
    ax = fig.add_subplot(1,3,1)
    im = ax.imshow(hyperGt,aspect='auto',cmap='jet',vmin=0,vmax=4)
    fig.colorbar(im)
    ax.set_xlabel('Ground Truth')
    ax = fig.add_subplot(1,3,2)
    im = ax.imshow(predic_map_include_zeroLabel, aspect='auto',cmap='jet',vmin=0,vmax=4)
    fig.colorbar(im)
    ax.set_xlabel('predict with zero label')
    ax = fig.add_subplot(1,3,3)
    im = ax.imshow(predic_map, aspect='auto',cmap='jet',vmin=0,vmax=4)
    fig.colorbar(im)
    ax.set_xlabel('predict without zero label')
    plt.close(fig)
    fig.savefig('dection.png',dpi=700)


def excute(folder_path, data_path ,gt_path):
    PATH = '/bjkangNet_add.pth'
    batch_size = 1
    test_sample = 0.3
    hyperCube, hyperGt, input_channels, n_classes, _, test_loader = load_mat_data.load_mat(data_path ,gt_path, batch_size, test_sample)


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = oneDCNN.bjkangNet(input_channels,n_classes)
    w_path = folder_path + PATH
    model.load_state_dict(torch.load(w_path))
    print(model)

    model.to(DEVICE)
    test_loss, test_accuracy = oneDCNN.evaluate(model, test_loader,0,DEVICE)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

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
                spectral = spectral.to(DEVICE)
                output = model(spectral)
                a, b = torch.max(output,1)
                predic_map[i,j] = b
                # if a >= 30:
                #     b = 0
                predic_map_include_zeroLabel[i, j] = b   
            
                if hyperGt[i,j] == 0:
                    predic_map[i, j] = 0


    fig = plt.figure(1,figsize=(13,4))
    ax = fig.add_subplot(1,3,1)
    im = ax.imshow(hyperGt,aspect='auto',cmap='jet',vmin=0,vmax=4)
    fig.colorbar(im)
    ax.set_xlabel('Ground Truth')
    ax = fig.add_subplot(1,3,2)
    im = ax.imshow(predic_map_include_zeroLabel, aspect='auto',cmap='jet',vmin=0,vmax=4)
    fig.colorbar(im)
    ax.set_xlabel('predict with zero label')
    ax = fig.add_subplot(1,3,3)
    im = ax.imshow(predic_map, aspect='auto',cmap='jet',vmin=0,vmax=4)
    fig.colorbar(im)
    ax.set_xlabel('predict without zero label')
    # plt.show()
    PATH = folder_path + '/dection.png'
    fig.savefig(PATH,dpi=700)
    plt.close(fig)
    return
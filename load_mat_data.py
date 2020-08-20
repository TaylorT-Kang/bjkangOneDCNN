from scipy import io
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def load_mat(path, gt_path,b_size=16,test_sample = 0.3):
    mat_file = io.loadmat(path)
    key_list = list(mat_file.keys())
    HSI_data = mat_file[key_list[3]]
    mat_file = io.loadmat(gt_path)
    key_list = list(mat_file.keys())
    HSI_gt = mat_file[key_list[3]]

    h, w, input_channels = HSI_data.shape
    n_classes = HSI_gt.max()
    hsi_data = []
    target = []
    k = 0
    # for i in range(h):
    #     for j in range(w):
    #         if HSI_gt[i,j] != 0:
    #             if HSI_gt[i,j] not in hsi_data:
    #                 hsi_data[HSI_gt[i,j]] = []
    #             hsi_data_i = np.expand_dims(HSI_data[i,j,:],axis=0)
    #             hsi_data[HSI_gt[i,j]].append(hsi_data_i)
    #             target.append(HSI_gt[i,j])

    for i in range(h):
        for j in range(w):
            if HSI_gt[i,j] != 0:
                hsi_data_i = np.expand_dims(HSI_data[i,j,:],axis=0)
                hsi_data.append(hsi_data_i)
                target.append(HSI_gt[i,j])
    # for key in hsi_data:

    x_train_tensor = torch.from_numpy(np.array(hsi_data,dtype=np.float)).float()
    y_train_tensor = torch.from_numpy(np.array(target,dtype=np.long)).long()

    X_train, X_test, y_train, y_test = train_test_split(x_train_tensor,y_train_tensor,test_size = test_sample, stratify=y_train_tensor)
    np.unique(y_train, return_counts=True)
    np.unique(y_test, return_counts=True)
    train_dataset = CustomDataset(X_train,y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=True)

    return input_channels, n_classes + 1 , train_loader, test_loader, 
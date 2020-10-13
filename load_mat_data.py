from scipy import io
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import hyperNomalize
import cv2

from scipy.ndimage import gaussian_filter1d
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        y = torch.from_numpy(self.y[idx]).int64()
        return x, y

    def __len__(self):
        return len(self.x)


def load_mat(path, gt_path, b_size=16, test_sample = 0.7):
    mat_file = io.loadmat(path)
    key_list = list(mat_file.keys())
    HSI_data = mat_file[key_list[3]]
    mat_file = io.loadmat(gt_path)
    key_list = list(mat_file.keys())
    HSI_gt = mat_file[key_list[3]]
    HSI_data = np.array(HSI_data, dtype=np.float)

    HSI_data = hyperNomalize.hyperNormalize(HSI_data)

    h, w, input_channels = HSI_data.shape
    n_classes = HSI_gt.max()
    hsi_data = []
    target = []
    k = 0

    # for i in range(h):
    #     for j in range(w):
    #         y = HSI_data[i,j,:]
    #         y_gau = gaussian_filter1d(y,1)
    #         HSI_data[i,j,:] = y_gau

    for i in range(h):
        for j in range(w):
            if HSI_gt[i,j] != 0:
                # if HSI_gt[i,j] == 4:
                #     continue
                hsi_data_i = np.expand_dims(HSI_data[i,j,:],axis=0)
                hsi_data.append(hsi_data_i)
                target.append(HSI_gt[i,j])
                # for k in range(20):
                #     hsi_data_i = np.expand_dims(HSI_data[i,j,:],axis=0)
                #     hsi_data.append(hsi_data_i)
                #     target.append(HSI_gt[i,j])

    X = torch.from_numpy(np.array(hsi_data,dtype=np.float)).float()
    y = torch.from_numpy(np.array(target,dtype=np.long)).long()

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_sample, stratify=y)

    total_dataset = TensorDataset(X, y)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    total_data_loader = DataLoader(dataset=total_dataset, batch_size=1, shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=True)

    return HSI_data, HSI_gt, input_channels, n_classes + 1 , train_loader, test_loader, total_data_loader
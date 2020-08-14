from scipy import io
import numpy as np
import torch
from torch.utils.data.dataset import random_split
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


def load_mat(path, gt_path,b_size=16,train_sample = 0.3):
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
    for i in range(h):
        for j in range(w):
            if HSI_gt[i,j] != 0:
                hsi_data_i = np.expand_dims(HSI_data[i,j,:],axis=0)
                hsi_data.append(hsi_data_i)
                target.append(HSI_gt[i,j])

    x_train_tensor = torch.from_numpy(np.array(hsi_data,dtype=np.float)).float()
    y_train_tensor = torch.from_numpy(np.array(target,dtype=np.long)).long()

    dataset = CustomDataset(x_train_tensor,y_train_tensor)
    total_length = int(dataset.__len__())
    train_length = int(total_length * 0.3)
    test_length = total_length - train_length

    train_dataset, test_dataset = random_split(dataset,[train_length, test_length])

    train_loader = DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=b_size, shuffle=True)

    return input_channels, n_classes + 1 , train_loader, test_loader, 
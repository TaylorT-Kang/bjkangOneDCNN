import oneDCNN
import detection
import BandSelection
import visualization
import time
import os
import load_mat_data
from scipy import io
import numpy as np

now = time.localtime()
date = str(now.tm_year) + '_ ' + str(now.tm_mon) + '_ ' + str(now.tm_mday) + '_ ' + str(now.tm_hour) + '_ ' + str(now.tm_min) + '_ ' + str(now.tm_sec)
path_folder = './Result/' + date
os.makedirs(path_folder)

data_path ,gt_path = './Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_GT.mat'
band_path = ' '

EPOCHS = 100
batch_size = 8
test_sample = 0.7
HSI_data, HSI_gt, input_channels, n_classes, train_loader, test_loader, total_loader = load_mat_data.load_mat(data_path, gt_path, batch_size, test_sample)


print('oneDCNN executed')
model_dict = oneDCNN.excute(path_folder, EPOCHS, batch_size, test_sample,input_channels, n_classes, train_loader, test_loader)

print('detection executed')
detection.excute(path_folder, HSI_data, HSI_gt,input_channels, n_classes, test_loader, model_dict)

mat_file = io.loadmat(band_path)
key_list = list(mat_file.keys())
band = mat_file[key_list[3]]
band = np.squeeze(band)

print('BandSelection executed')
BandSelection.excute(path_folder, input_channels, n_classes, total_loader, test_loader, band, model_dict)

label_name = { }


print('visualization executed')
visualization.excute(path_folder, label_name)
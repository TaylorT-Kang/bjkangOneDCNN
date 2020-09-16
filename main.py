import oneDCNN
import detection
import BandSelection
import visualization
import time
import os

now = time.localtime()
date = str(now.tm_year) + '_ ' + str(now.tm_mon) + '_ ' + str(now.tm_mday) + '_ ' + str(now.tm_hour) + '_ ' + str(now.tm_min) + '_ ' + str(now.tm_sec)
path_folder = './Result/' + date
os.makedirs(path_folder)


data_path ,gt_path = './Datasets/PaviaU/PaviaU.mat', './Datasets/PaviaU/PaviaU_gt.mat'

EPOCHS = 400
batch_size = 100
test_sample = 0.7

print('oneDCNN executed')
oneDCNN.excute(path_folder,data_path, gt_path, EPOCHS, batch_size, test_sample)

print('detection executed')
detection.excute(path_folder, data_path ,gt_path)

print('BandSelection executed')
BandSelection.excute(path_folder, data_path ,gt_path)

print('visualization executed')
visualization.excute(path_folder)
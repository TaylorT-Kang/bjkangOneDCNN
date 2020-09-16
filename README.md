bjkangOneDCNN

+ 1-dimension CNN to classify hyperspectral images
  + Reference[1] - Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks, Chen et al., TGRS 2016
  + Reference[2] - Band Selection via Explanations From Convolutional Neural Networks, L. Zhao, Y. Zeng, P. Liu and G. He, IEEE Access 2020

+ Load_mat.py
  + It is to load .mat file, the file consists of W x H x Channels.

+ Apply Grad-CAM
  + It is being appied by refering to "https://github.com/jacobgil/pytorch-grad-cam.git"

+ How to use
  + Install packages about 'requirment.txt'
  + In 'main.py', set the data path.
  + In shell, execute 'main.py'
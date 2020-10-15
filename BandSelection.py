import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gradcam
import oneDCNN
import load_mat_data
from scipy import io


def show_bandSelection(avg_gradcam,fig, band):
    number_of_key = len(avg_gradcam)
    cols = 1
    rows = number_of_key
    i = 1
    for key, val in avg_gradcam.items():
        ax = fig.add_subplot(rows,cols,i)
        # start = float("{:.2f}".format(round(band[0],2)))
        # end = float("{:.2f}".format(band[272]))
        # _, length = val.shape
        ax.imshow(val,aspect='auto')
        ax.set_xticklabels(band)
        ax.set_ylabel(key)
        ax.set_xlabel('um')
        i += 1
        
    # plt.show()
    return

if __name__=='__main__':

    PATH = './bjkangNet_add.pth'
    batch_size = 1
    test_sample = 0.3
    _, _, input_channels, n_classes, train_loader, test_loader = load_mat_data.load_mat('./ADD_data/PaintsSameLabel/hyper3D_MWIR.mat', './ADD_data/PaintsSameLabel/hyper3D_MWIR_GT.mat',batch_size, test_sample)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = oneDCNN.bjkangNet(input_channels,n_classes)
    model.load_state_dict(torch.load(PATH))
    print(model)
    model.to(DEVICE)
    test_loss, test_accuracy = oneDCNN.evaluate(model, test_loader,0,DEVICE)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

    grad_cam = gradcam.GradCam(model=model, feature_module=model, target_layer_names=["conv5"], use_cuda=DEVICE)
    gb_model = gradcam.GuidedBackpropReLUModel(model=model, use_cuda=DEVICE)
    # test_spectrals, labels = iter(test_loader).next()
    # input_ = test_spectrals.requires_grad_(True)
    # mask = grad_cam(input_,None)
    # gb = gb_model(test_spectrals,index=None)

    gb_list = {}
    grad_cam_list = {}

    dataiter = iter(train_loader)
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

    path = './ADD_data/PaintsSameLabel/MWIR_band.mat'
    mat_file = io.loadmat(path)
    key_list = list(mat_file.keys())
    band = mat_file[key_list[3]]
    band = np.squeeze(band)

    fig = plt.figure(1)
    show_bandSelection(guided_grad_cam, fig, band)

    fig.savefig('band.png',dpi=300)

    import pickle

    with open('SelectionBand.pickle','wb') as f:
        pickle.dump(guided_grad_cam,f)
        pickle.dump(band,f)

def excute(path_folder, input_channels, n_classes, total_loader, test_loader, band, model_dict):


    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = oneDCNN.bjkangNet(input_channels,n_classes)
    model.load_state_dict(model_dict)
    print(model)
    model.to(DEVICE)
    test_loss, test_accuracy = oneDCNN.evaluate(model, test_loader,0,DEVICE)
    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss, test_accuracy))

    grad_cam = gradcam.GradCam(model=model, feature_module=model, target_layer_names=["conv5"], use_cuda=DEVICE)
    gb_model = gradcam.GuidedBackpropReLUModel(model=model, use_cuda=DEVICE)
    # test_spectrals, labels = iter(test_loader).next()
    # input_ = test_spectrals.requires_grad_(True)
    # mask = grad_cam(input_,None)
    # gb = gb_model(test_spectrals,index=None)

    gb_list = {}
    grad_cam_list = {}

    dataiter = iter(total_loader)
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

    

    fig = plt.figure(1)
    show_bandSelection(guided_grad_cam, fig, band)
    
    PATH = path_folder + '/band_heatmap.png'
    fig.savefig(PATH, dpi=300)
    plt.close(fig)
    import pickle
    pickle_path = path_folder + '/SelectionBand.pickle'
    with open(pickle_path, 'wb') as f:
        pickle.dump(guided_grad_cam,f)
        pickle.dump(band,f)

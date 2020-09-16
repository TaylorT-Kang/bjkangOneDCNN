import pickle
import matplotlib.pyplot as plt
import numpy as np

def show_bandSelection(avg_gradcam, fig, band, label_name):
    number_of_key = len(avg_gradcam)
    cols = 1
    rows = number_of_key
    i = 1
    # band = np.round(band[:],1)
    for key, val in avg_gradcam.items():
        val = np.squeeze(val)
        ax = fig.add_subplot(rows,cols,i)
        ax.stem(band,val,use_line_collection = True)
        ax.set_ylabel(label_name[key])
        ax.set_xlabel('um')
        i += 1

    # plt.show()
    return

if __name__=='__main__':

    with open('SelectionBand.pickle','rb') as f:
        bandSelection = pickle.load(f)
        band = pickle.load(f)


    for key, val in bandSelection.items():
        minVal = np.min(val)
        maxVal = np.max(val)

        normalizeM = bandSelection[key] - minVal
        if maxVal == minVal:
            normalizeM = np.zeros(val.shape)
        else:
            normalizeM = normalizeM / (maxVal - minVal)

        bandSelection[key] = normalizeM
    label_name = {1 : 'paint', 2 : 'aspalt', 3 : 'grass', 4 : 'black body'}
    fig = plt.figure(1,figsize=(20,10))
    show_bandSelection(bandSelection, fig, band, label_name)
    fig.savefig('band.png',dpi=300)

def excute(folder_path):
    pickle_path = folder_path + '/SelectionBand.pickle'
    with open(pickle_path,'rb') as f:
        bandSelection = pickle.load(f)
        band = pickle.load(f)

    for key, val in bandSelection.items():
        minVal = np.min(val)
        maxVal = np.max(val)

        normalizeM = bandSelection[key] - minVal
        if maxVal == minVal:
            normalizeM = np.zeros(val.shape)
        else:
            normalizeM = normalizeM / (maxVal - minVal)

        bandSelection[key] = normalizeM
    
    label_name = {1 : 'Asphalt', 2 : 'Meadows', 3 : 'Gravel', 4 : 'Trees', 5 : 'Painted metal sheets', 6 : 'Bare Soil', 7 : 'Bitumen', 8 : 'Self-Blocking Bricks', 9 : 'Shadows'}
    fig = plt.figure(1,figsize=(20,10))
    show_bandSelection(bandSelection, fig, band, label_name)
    PATH = folder_path + '/band.png'
    fig.savefig(PATH, dpi=700)
    plt.close(fig)
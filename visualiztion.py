import pickle
import matplotlib.pyplot as plt

with open('SelectionBand.pickle','rb') as f:
    bandSelection = pickle.load(f)

def show_bandSelection(avg_gradcam,fig):
    number_of_key = len(avg_gradcam)
    fig = plt.figure(1)
    cols = 1
    rows = number_of_key
    i = 1
    for key, val in avg_gradcam.items():
        ax = fig.add_subplot(rows,cols,i)
        ax.imshow(val,aspect='auto')
        ax.set_ylabel(key)
        i += 1

    plt.show()
    return

fig = plt.figure(1)
# show_bandSelection(bandSelection,fig)


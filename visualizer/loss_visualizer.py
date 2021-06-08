# import pandas as pd
# import torch
import matplotlib.pyplot as plt
import numpy as np
def visualizeLosses():
    # pdFile = pd.read_csv('saved_figs/track_lossImg.csv')
    data = np.load('track_lossImg.npy')
    
    data -= np.min(data)
    # print(np.max(data), np.min(data), np.mean(data))

    # cont = np.greater(data, np.mean(data) + 2*np.std(data))
    # data[cont] = np.mean(data) + 2*np.std(data)
    # data = 255 * data
    data = data.astype(int)
    # thisData = data[2,:,:]
    data2 = np.load('track_lossNum.npy')

    for i in range(data2.shape[1]):
        plt.plot(data2[:,i],label='Image '+str(i))
    plt.legend()

    # plt.figure()
    # print(np.max(thisData), np.min(thisData), np.mean(thisData))
    # print(thisData.shape)
    # plt.imshow(thisData)
    # plt.show()


    class IndexTracker:
        def __init__(self, ax, X):
            self.ax = ax
            ax.set_title('use scroll wheel to navigate images')

            self.X = X
            self.slices, rows, cols = X.shape
            self.ind = 1
            self.im = ax.imshow(self.X[self.ind,:, :])
            self.update()

        def on_scroll(self, event):
            # print("%s" % (event.key))
            # print(self.slices)
            if event.key == 'right':
                if (self.ind + 1) < self.slices:
                    self.ind = (self.ind + 1) % self.slices
            elif event.key == 'left':
                if (self.ind - 1) > 0:
                    self.ind = (self.ind - 1) % self.slices
            else:
                print(event.key)
            self.update()

        def update(self):
            self.im.set_data(self.X[self.ind,:, :])
            self.ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()


    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, data)

    fig.canvas.mpl_connect('key_press_event', tracker.on_scroll)
    plt.show()
    # print(pdFile.iloc[0,1])
    # plt.show()
    # print(pdFile)
visualizeLosses()

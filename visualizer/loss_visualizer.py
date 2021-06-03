# import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
def visualizeLosses():
	# pdFile = pd.read_csv('saved_figs/track_lossImg.csv')
	data = np.load('saved_figs/track_lossImg.npy')
	data = 255 * data
	data = data.astype(int)
	data2 = np.load('saved_figs/track_lossNum.npy')
	print(np.max(data), np.min(data))
	print(data.shape)
	print(data2[1,:])
	plt.imshow(data[1,:,:])
	plt.savefig('test.png')

	# print(pdFile.iloc[0,1])
	# plt.show()
	# print(pdFile)
visualizeLosses()
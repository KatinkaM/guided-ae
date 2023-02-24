import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def plot_code_image(abu_image):
    abu = "C:/Users/katin/Documents/NTNU/Semester_10/data/Gabor_data/"+abu_image + "SSIIFDv2s.mat"

    gabor = sio.loadmat(abu)['score_SSIIFD']
    gabor = gabor.reshape(100,100,-1)
    plt.imshow(gabor)
    plt.show()



abu = "abu-airport-1"
plot_code_image(abu)

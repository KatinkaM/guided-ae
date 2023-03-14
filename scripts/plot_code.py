import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

a = "abu-urban-5"


AE = "normalNet"+a+ ".mat"
Gabor = "gab" + a + ".mat"

abu = "C:/Users/katin/Documents/NTNU/Semester_10/data/ABU_data/abu-airport-1.mat"
AUC_list = []

code_AE = np.transpose(sio.loadmat(AE)['code'])
code_Gabor = np.transpose(sio.loadmat(Gabor)['code'])

det_AE = np.transpose(sio.loadmat(AE)['detection'])
det_Gabor = np.transpose(sio.loadmat(Gabor)['detection'])

back_AE = np.transpose(sio.loadmat(AE)['background'])
back_Gabor = np.transpose(sio.loadmat(Gabor)['background'])

# fig, axs = plt.subplots(nrows=1, ncols=2)

# AUC_GAB = sio.loadmat(Gabor)['auc_scores']
# AUC_Norm = sio.loadmat(AE)['auc_scores']

# # Plot matrix a in the first subplot
# axs[0].plot(AUC_Norm[0])
# axs[0].set_title('Linear AUC')

# # Plot matrix b in the second subplot
# axs[1].plot(AUC_GAB[0])
# axs[1].set_title('Gabor AUC')
# plt.show()

fig, axs = plt.subplots(nrows=3, ncols=2)

# Plot matrix a in the first subplot
axs[0,0].imshow(code_AE[:,:,0], cmap='jet')
axs[0,0].set_title('Linear conv code')

# Plot matrix b in the second subplot
axs[0,1].imshow(code_Gabor[:,:,0], cmap='jet')
axs[0,1].set_title('Gabor conv code')

axs[1,0].imshow(det_AE, cmap='jet')
axs[1,0].set_title('Linear conv det')

# Plot matrix b in the second subplot
axs[1,1].imshow(det_Gabor, cmap='jet')
axs[1,1].set_title('Gabor conv det')

axs[2,0].imshow(back_AE[:,:,0], cmap='jet')
axs[2,0].set_title('Linear conv background')

# Plot matrix b in the second subplot
axs[2,1].imshow(back_Gabor[:,:,0], cmap='jet')
axs[2,1].set_title('Gabor conv background')



for ax in axs.flat:
    ax.axis('off')

plt.show()
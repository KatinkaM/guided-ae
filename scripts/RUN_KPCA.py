
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os 

data_path = "C:/Users/katin/Documents/NTNU/Semester_10/AUTO-AD-test/"
residual_path = "./test_KPCA/abu-airport-1.mat"
img = np.array(sio.loadmat(os.path.join(data_path, "abu-airport-1.mat"))['data']).real
img_reshape = img.reshape(img.shape[0]*img.shape[1],-1)[:,:100]#Only use the first 100 dimensions
img_n = MinMaxScaler(feature_range = (-1,1)).fit_transform(img_reshape)
transformer = KernelPCA(n_components=100, kernel='rbf')
X_transformed = transformer.fit_transform(img_n)
a = X_transformed.reshape(100,100,-1)
plt.imshow(a[:,:,0])
plt.show()
sio.savemat(residual_path, {'abu': X_transformed})
X_transformed.shape
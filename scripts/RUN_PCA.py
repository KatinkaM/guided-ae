from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os 
import time

data_path = "C:/Users/katin/Documents/NTNU/Semester_10/AUTO-AD-test/data_no_preprocessing/"
abu_list = ["abu-airport-1","abu-airport-2","abu-airport-3","abu-airport-4","abu-beach-1","abu-beach-2", "abu-beach-3","abu-beach-4", "abu-urban-1", "abu-urban-2", "abu-urban-3", "abu-urban-4", "abu-urban-5"]

for i in range(len(abu_list)):
    start = time.time()
    abu = abu_list[i]
    print(abu)
    abu_name = abu + ".mat"
    residual_path = "./test_PCA/"+abu+"PCA_nomm.mat"
    img = np.array(sio.loadmat(os.path.join(data_path, abu_name))['data']).real
    img_reshape = img.reshape(img.shape[0]*img.shape[1],-1)
    #img_n = MinMaxScaler(feature_range = (-1,1)).fit_transform(img_reshape)
    transformer = PCA(n_components=100)
    X_transformed = transformer.fit_transform(img_reshape)
    end = time.time()
    print(end-start)
    sio.savemat(residual_path, {'abu': X_transformed})

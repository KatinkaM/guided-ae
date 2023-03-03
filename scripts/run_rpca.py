import numpy as np
from r_pca import R_pca
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler

# generate low rank synthetic data
N = 100
num_groups = 3
num_values_per_group = 40
p_missing = 0.2
abu_list = [ "abu-urban-2"]#, "abu-urban-3", "abu-urban-4", "abu-urban-5"]#["abu-airport-3","abu-beach-1","abu-beach-2", "abu-beach-3", "abu-beach-4","abu-urban-1", "abu-urban-2", "abu-urban-3", "abu-urban-4", "abu-urban-5"]

for i in range(len(abu_list)):
    HSI_img = abu_list[i]
    HSI_mat_file = HSI_img + ".mat"

    HSI_path = "Dataset/" + HSI_mat_file
    ##Loading HSI
    X_raw = sio.loadmat(HSI_path)['data']

    D = X_raw.reshape(X_raw.shape[0]*X_raw.shape[1],-1)

    # use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
    rpca = R_pca(D)
    L, S = rpca.fit(max_iter=1000, iter_print=100)

    L_reshape = L.reshape(X_raw.shape[0],X_raw.shape[1],-1)
    save_path = "./RPCA_plots/" + HSI_img + "L.mat"
    sio.savemat(save_path,{'abu': L_reshape})

    S_reshape = S.reshape(X_raw.shape[0],X_raw.shape[1],-1)
    save_path = "./RPCA_plots/" + HSI_img + "S.mat"
    sio.savemat(save_path,{'abu': S_reshape})

# visually inspect results (requires matplotlib)
# rpca.plot_fit()
# plt.show()
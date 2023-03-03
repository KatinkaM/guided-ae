import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


abu = "abu-urban-3mkpca"

det = sio.loadmat(abu)['map']
det = det.reshape(100,100)
plt.figure(figsize=(5,5))
plt.axis('off')
plt.imshow(det)
plt.savefig(abu,bbox_inches='tight')
plt.show()


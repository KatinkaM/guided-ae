import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


abu = "detection_testing.mat"

det = sio.loadmat(abu)['detection']
det = det.reshape(100,100)
plt.figure(figsize=(5,5))
plt.axis('off')
plt.imshow(det)
plt.savefig(abu,bbox_inches='tight')
plt.show()


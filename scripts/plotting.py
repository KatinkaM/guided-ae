import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


fig, axs = plt.subplots(nrows=4, ncols=4,figsize=(10, 10))
abu_list =["abu-airport-1","abu-airport-2","abu-airport-3","abu-airport-4","abu-beach-1","abu-beach-2", "abu-beach-3", "abu-beach-4","abu-urban-1", "abu-urban-2", "abu-urban-3", "abu-urban-4", "abu-urban-5"]

matrices = []
for i in range(len(abu_list)):
  abu ="detection_testing"+ abu_list[i] + ".mat"
  pic = sio.loadmat(abu)

  code = pic['code32'].sum(0)
#   matrices.append(np.array(code.sum(0)))
  det = pic['detection']
  c = 0.2*code + 0.8*det
  matrices.append(np.array(c))
  

  # auc = np.transpose(pic['auc_score'])
  # plt.plot(auc)
  # plt.show()

# Iterate over the axes of the subplot
for i, ax in enumerate(axs.flat):

    # Plot the i-th matrix on the current axis
    if i < len(matrices):
        ax.imshow(matrices[i])
        ax.axis('off')
    else:
        # If there are no more matrices to plot, hide the axis
        ax.axis('off')

# Adjust the spacing between subplots to make them fit better
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# Show the plot
plt.show()
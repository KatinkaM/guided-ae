import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


fig, axs = plt.subplots(nrows=4, ncols=4,figsize=(10, 10))
abu_list =["abu-airport-1","abu-airport-2","abu-airport-3","abu-airport-4","abu-beach-1","abu-beach-2", "abu-beach-3", "abu-beach-4","abu-urban-1", "abu-urban-2", "abu-urban-3", "abu-urban-4", "abu-urban-5"]

matrices = []
for i in range(len(abu_list)):
  abu = abu_list[i] + "S.mat"
  gt_path = "C:/Users/katin/Documents/NTNU/Semester_10/data/RPCA_data/"+ abu
  pic = sio.loadmat(gt_path)

  det = pic['abu']
  matrices.append(np.array(det[:,:,0]))
  

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
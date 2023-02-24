import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def plot_code_image(image, abu_image):
    abu = "C:/Users/katin/Documents/NTNU/Semester_10/data/ABU_data/"+abu_image + ".mat"

    det = sio.loadmat(image)['detection']

    
    #Calculatin AUC score 
    gt = sio.loadmat(abu)['map'][:100,:100] #When using KPCA I have to do this!!!!

    back = sio.loadmat( image)['background']
    back = back[0,:,:]
    print(back.shape)


    code_32 = sio.loadmat( image)['code32']
    
    code_32 = code_32[0,:,:]

    code_60 = sio.loadmat( image)['code60']
    code_60 = code_60[0,:,:]

    code_62 = sio.loadmat( image)['code62']
    code_62 = code_62[0,:,:]
    
    code_30 = sio.loadmat( image)['code30']
    code_30 = code_30[0,:,:]
  
    
    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs[0, 0].imshow(back)
    axs[0, 0].set_title('Background')

    # plot on the second subplot (top right)
    axs[0, 1].imshow(det)
    axs[0, 1].set_title('Detection Map')

    # plot on the third subplot (bottom left)
    axs[1, 0].imshow(code_32)
    axs[1, 0].set_title('Image Code')

    # plot on the fourth subplot (bottom right)
    axs[1, 1].imshow(code_30)
    axs[1, 1].set_title('Block 30')

    axs[0, 2].imshow(code_60)
    axs[0, 2].set_title('Block 60')

    # plot on the fourth subplot (bottom right)
    axs[1, 2].imshow(code_62)
    axs[1, 2].set_title('Block 62')

  
    plt.show()


residual_root_path = "./results/detection_testing"
abu = "abu-beach-2"
plot_code_image(residual_root_path,abu)

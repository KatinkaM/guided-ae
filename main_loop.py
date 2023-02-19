import numpy as np 
import torch
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
import os
from utils.common import skip,features
from utils.common_utils import get_noise, get_params
import time

from scripts.calculate_results import calculate_AUC


data_path = "C:/Users/katin/Documents/NTNU/Semester_10/AUTO-AD-test/data/KPCA_data"
dtype = torch.FloatTensor

residual_root_path = "./results/detection_testing_2"

torch.autograd.set_detect_anomaly(True)


def main(abu):
    start = time.time()
    # data input
    # **************************************************************************************************************
    #torch.cuda.empty_cache()
    
    background_root_path = "./background"
    thres = 0.000015
    chanels = 32
    layers = 3
    abu_path = abu + ".mat"
    
    print(data_path)
  

  
    img = np.array(sio.loadmat(os.path.join(data_path, abu_path))['abu'],dtype=np.float16) #.real

    img_reshape = img.reshape(img.shape[0]*img.shape[1],-1)[:,:100]#Only use the first 100 dimensions

    img_n = MinMaxScaler(feature_range = (0,1)).fit_transform(img_reshape)
    
    img_processed = np.reshape(img_n,(img.shape[0],img.shape[1],-1))

    #Transpose to get the correct dimesnison for the cnn
    img_np = np.transpose(img_processed)
 
    #Creating tensors from the numpy.ndarray
    img_var = torch.from_numpy(img_np).type(dtype)

    img_size = img_var.shape
    #Retrevieng number of bands, rows and colunms
    band = img_size[0]
    row = img_size[1]
    col = img_size[2]

    # model setup
    # **************************************************************************************************************
    pad = 'reflection' #'zero' and reflection gives padding when k = 3 (p =1), need this to be able to run the code
    OPT_OVER = 'net'
    # OPTIMIZER = 'adam'
    method = '2D'
    input_depth = img_np.shape[0] #Number of bands
    output_depth = img_np.shape[0] #Number of bands
    LR = 0.01
    num_iter = 1001
    param_noise = False
    reg_noise_std = 0.1 # 0 0.01 0.03 0.05
    #skip from models library file - > should assemble encoder-decoder with skip connects
    net = skip(input_depth, output_depth,
              num_channels_down = [chanels] * layers,
              num_channels_up =   [chanels] * layers,
              num_channels_skip =    [chanels] * layers,
              filter_size_up = 1, filter_size_down = 1,
              upsample_mode='nearest', filter_skip_size=1,
              need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    net = net.type(dtype) # see network structure
    
    net_input = get_noise(input_depth, method, img_np.shape[1:]).type(dtype) #Input with input_depth as nr of channels, method is now set to 
    # #2D and it tells us which noise should be used to fill tensor (common_utils) -> outputs Tensor with uniform noise

    #sums up amount of parameters
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = img_var[None, :]#.cuda()#Enables GPU on tensor of HSI actual image

    mask_var = torch.ones(1, band, row, col)#.cuda() #Tensor with ones GPU enabled [1,162,80,100], used to hold the reconstructed background
    residual_varr = torch.ones(row, col)#.cuda() #Tensor with ones [80,100], used to hold the detected anomalies

    

    def closure(iter_num, mask_varr, residual_varr):

        if param_noise: #This is false
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0: #This is 0.1
            net_input = net_input_saved + (noise.normal_() * reg_noise_std) #Input HSI NOISE is changed with the reg_noise for every iteration
        #noise.normal_() fills noise with normal noise*reg, net_input is the same for the first iteration

        out = net(net_input) #Input is the HSI NOISE with added reg_noise, and it is run trough the split model net
        out_np = out.detach().cpu().squeeze().numpy() #Returns a tensor detached from the graph, retrurns copy to cpu, squeeze removes dim of size 1
        #numpy converts to numpy array

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 100==0 and iter_num!=0:
            # weighting block
            img_var_clone = img_var.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :]) #(output-HSI)^2 reconstruction error eq (8)
            residual_img = temp.sum(0) #create the map of the reconstruction errors E (9)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            # residuals to weights
            residual_img = r_max - residual_img #Residual image eq (10,11)
            r_min, r_max = residual_img.min(), residual_img.max() #Retrieve min and max value of the residual weights. The smallest indicate anomalies
            residual_img = (residual_img - r_min) / (r_max - r_min) #redusing the contributions of the anomalies 

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):#iterate trough the resiudal image and add in mask_var_clone which is the variance 
                mask_var_clone[0, i, :] = residual_img[:]

        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone) #mask_var is the detected background
        total_loss.backward() #Backprop
        #print("iteration: %d; loss: %f" % (iter_num+1, total_loss))

        return mask_var_clone, residual_var_clone, out_np, total_loss

    net_input_saved = net_input.detach().clone() #Copy the HSI with ONLY NOISE
    noise = net_input.detach().clone() #Copy the HSI with ONLY NOISE
    loss_np = np.zeros((1, 50), dtype=np.float32) #Loss is vector of size 50
    loss_last = 0
    end_iter = False


    p = get_params(OPT_OVER, net, net_input) #In common utils:
    #Optimize over net (which is the split network ) and the tensor that stores the tinput is net_input (ONLY NOISE)
    #Returns the parameters in the net network (split)
    #print('Starting optimization with ADAM')

    optimizer = torch.optim.Adam(p, lr=LR)
    
    
   
    for j in range(num_iter): #iterate 1001 times
        optimizer.zero_grad() #Sets gradients of all optimizers to zero

        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step() #Updates the parameters based on the optimizer

        if j >= 1:
            index = j-int(j/50)*50 
            loss_np[0][index-1] = abs(loss-loss_last)
            if j % 50 == 0: #Check if number is dividable by 50, if so we check if loss is below a certain value. this is early stop algorithm
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss
       
        if j == num_iter-1 or end_iter == True: #Happens if iterations = 1000 or end it is True
            print("Number of iterations: " + str(j))
            residual_np = residual_varr.detach().cpu().squeeze().numpy() #resiudal variance
            residual_path = residual_root_path + ".mat" #Go into the detection image
            end = time.time()
            print("Run time: ", end-start)
            #Retrieving value of the code
            code = features["B32"].detach().cpu().squeeze().numpy() 
            code_2 = features["B60"].detach().cpu().squeeze().numpy() 

            sio.savemat(residual_path, {'detection': residual_np,'time': end-start, 'code': code, 'code2' :code_2}) #save residual variance to this image
            

            background_path = background_root_path + ".mat"
            sio.savemat(background_path, {'detection': background_img.transpose(1, 2, 0)}) #Save background image which is the output of the network
            return

if __name__ == "__main__":
    main("abu-airport-1")
    calculate_AUC(residual_root_path)

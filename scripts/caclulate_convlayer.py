import numpy as np
def calc_output_size(w1,k,s,p=0):
    return np.floor((w1+2*p-(k))/s)+1

print(calc_output_size(13,3,2,1))
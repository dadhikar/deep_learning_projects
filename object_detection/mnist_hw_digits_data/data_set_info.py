"""Created on Tue Jun  9 10:19:19 2020 @author: dadhikar"""

import os
import sys
import h5py
import matplotlib.pyplot as plt 

# specify data file location
w_dir = os.getcwd()
print(w_dir)
sys.exit()
data_path = w_dir + os.sep + "data_set"
for file in os.listdir(data_path):
    print(file)
    
train_file = h5py.File(data_path+os.sep+ "train_cat_vs_noncat.h5", 'r')    
test_file = h5py.File(data_path+os.sep+ "test_cat_vs_noncat.h5", 'r')


print('-'*10, "Train set data", '-'*10 )
train_file_info = [train_file.keys()] 
print("File info:", train_file_info)
train_set_X = train_file["train_set_x"]
train_set_y = train_file["train_set_y"]
print("Train set X data: shape:", train_set_X.shape)
print("Train set y data: shape:", train_set_y.shape)
print("Train set X data: type:", train_set_X.dtype)
print("Train set y data: type:", train_set_y.dtype)
print('-'*40)



print('-'*10, "Test set data", '-'*10 )
test_file_info = [test_file.keys()] 
print("File info:", test_file_info)
test_set_X = test_file["test_set_x"]
test_set_y = test_file["test_set_y"]
print("Test set X data: shape:", test_set_X.shape)
print("Test set y data: shape:", test_set_y.shape)
print("Test set X data: type:", test_set_X.dtype)
print("Test set y data: type:", test_set_y.dtype)
print('-'*40)


# plt.imshow(train_set_X[3,:,:,:,])
# plt.axis('off')
# plt.show()

set_plot = False
if set_plot:
    nx = 7
    ny = 7
    fig, ax = plt.subplots(nx, ny)
    ax = ax.flatten()
    for i in range(len(ax)):
        ax[i].imshow(test_set_X[i,:,:,:,], alpha=1)
        # ax[i].set_title(r"{}". format(i+1))
        # ax[i].legend(loc='best')
        ax[i].axis('off')
       
    plt.show()    

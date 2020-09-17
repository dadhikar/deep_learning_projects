"""Created on Tue Jun  9 17:29:22 2020 @author: dadhikar"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

import deep_L_layer_neuron_networks as dlnn
import gradient_checking as grad_check

# specify data file location
w_dir = os.getcwd()
data_path = w_dir + os.sep + "data_set"
for file in os.listdir(data_path):
    print('data files:', file)
 
# reading h5 date file
train_file = h5py.File(data_path+os.sep+ "train_cat_vs_noncat.h5", 'r')    
test_file = h5py.File(data_path+os.sep+ "test_cat_vs_noncat.h5", 'r')

print('-'*30)
train_X = train_file["train_set_x"]
test_X = test_file["test_set_x"]
print('train_X shape:', train_X.shape)
print('test_X shape:', test_X.shape)
train_y = train_file["train_set_y"]
test_y = test_file["test_set_y"]
print('train_y shape:', train_y.shape)
print('test_y shape:', test_y.shape)
print('-'*30)

# coverting X into shape of (n_input_feature, n_samples)
train_X = np.asarray(train_X)
train_X = train_X.reshape(train_X.shape[0], -1).T
print('train_X shape:', train_X.shape)
test_X = np.asarray(test_X)
test_X = test_X.reshape(test_X.shape[0], -1).T
print('test_X shape:', test_X.shape)

# converting y into shape of (class_level, n_sample)
train_y = np.asarray(train_y) 
train_y = train_y.reshape(train_y.shape[0], 1).T
print('train_y shape:', train_y.shape)
test_y = np.asarray(test_y) 
test_y = test_y.reshape(test_y.shape[0], 1).T
print('test_y shape:', test_y.shape)
print('-'*30)

# Standardize data to have feature values between 0 and 1.
train_X = train_X/train_X.max()
test_X = test_X/test_X.max()



def deep_L_layer_nn(X, Y, layers_dims, learning_rate=0.0075, 
                    num_iterations=3000):
    
    """
    Implements a L-layer neural network:
    
    Arguments:
    X -- feature matrix of shape (n_features, n_samples)
    Y -- traget variable matrix of shape (1, n_samples) 
    layers_dims -- (n_input_node, n_hidden_nodes (nh1, nh2, ..), n_output_node)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. 
    """
    np.random.seed(1)
    # initialize parameters 
    parameters = dlnn.parameters_initialize(layers_dims)
    
    costs = []                         
    
    for i in range(0, num_iterations):
        # forward propagation
        AL, caches = dlnn.forward_propagation(X, parameters)
        
        # compute cost of each iteration
        cost = dlnn.compute_cost(AL, Y)
      
        
        # backpropagation
        grads = dlnn.back_propagation(AL, Y, caches)
        
        # updating parameters
        parameters = dlnn.update_parameters(parameters, grads, learning_rate)
        
        # printing cost function after every 100 iterations
        if i % 100 ==0:
            print('Cost after {} iteration'.format(i), np.squeeze(cost))
            costs.append(np.squeeze(cost))
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
             
    return costs, parameters, grads   
        


# training the network
layers_dims = [train_X.shape[0], 7, 5, 1] #  2-layer model
num_iterations = 101
learning_rate = 0.0075

_, parameters, grads = deep_L_layer_nn(train_X, train_y, layers_dims=layers_dims,
                                   num_iterations=num_iterations,
                                   learning_rate=learning_rate)


theta_parameters = grad_check.dictionary_to_vector(parameters)

theta_grads = grad_check.gradients_to_vector(grads)

num_parameters = theta_parameters.shape[0]

    
J_plus = np.zeros((num_parameters, 1))
J_minus = np.zeros((num_parameters, 1))
grad_approx = np.zeros((num_parameters, 1))
  
#sys.exit()
epsilon = 1e-5
for i in range(num_parameters):
        thetaplus = np.copy(theta_parameters)
        print('Before', thetaplus[i])
        thetaplus[i] +=  epsilon
        print('After', thetaplus[i])
        
        #parameters_ = grad_check.vector_to_dictionary(thetaplus, parameters)
        #print(parameters_)        
        #AL, _ = dlnn.forward_propagation(X, _parameters)                        
        #J_plus[i]= dlnn.compute_cost(AL, Y)
        
        
        #thetaminus = np.copy(theta_parameters)
        #thetaminus[i][0] = thetaminus[i][0] - epsilon
        #_parameters1 = dlnn.grad_check.vector_to_dictionary(thetaminus,
        #                                               parameters)
        #AL, _ = dlnn.forward_propagation(X, _parameters1)                        
        #J_minus[i] = dlnn.compute_cost(AL, Y)
        
        #gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
        # print(len(J_minus))
        #denominator = np.linalg.norm(theta_grad) + np.linalg.norm(gradapprox)              
        #numerator = np.linalg.norm((theta_grad-gradapprox))                                   
        #difference = numerator/denominator                                                  
    
        #print(difference)     
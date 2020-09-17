"""Created on Mon Jun  8 13:30:48 2020 @author: dadhikar"""

import numpy  as np 



def initialize_param(layers_dims):
    """
    Parameters
    ----------
    layers_dim : array (or list) of layers with node size

    Returns
    -------
    parameters: python dictionary with W1, b1, W2, B2, ....., WL, BL

    """
    parameters = {}
    np.random.seed(3)
    L = len(layers_dims)   # number of layers in the Network
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],
                                                 layers_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))
        
        assert parameters['W'+str(l)].shape == (layers_dims[l], layers_dims[l-1])
        assert parameters['b'+str(l)].shape ==(layers_dims[l], 1)
    return parameters

def linear_forward(A, W, b):
    # calculate net input 
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache





def activation_forward(A_prev, W, b, activation):
    
    
    # caculating activation for layer l
    if activation =='sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activate_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activate_cache = relu(Z)
    
    assert A.shape == Z.shape
    
    cache = (linear_cache, activate_cache)
    
    return A, cache 


def deep_L_layer_forward(X, parameters):
    
    caches = []
    L = len(parameters) //2
    A = X
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, parameters['W'+str(l)], 
                                      parameters['b'+str(l)], 'relu')
        caches.append(cache)

    AL, cache = activation_forward(A, parameters['W'+str(L)], 
                                   parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def sigmoid(Z):
    A = 1./(1+np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    a = 1./(1+np.exp(-Z))
    dZ = dA*a*(1-a)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    
    assert A.shape == Z.shape
    
    cache = Z
    return A, cache

    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    
    assert dZ.shape == Z.shape
    
    return dZ



def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = (-1/m)* np.sum((np.multiply(Y, np.log(AL)) + 
                           np.multiply((1-Y),np.log(1-AL))), keepdims=True)
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache 
    m = A_prev.shape[1]
    
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    assert dW.shape == W.shape
    assert db.shape == b.shape
    assert dA_prev.shape == A_prev.shape
    
    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):
    
    linear_cache, activate_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activate_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activate_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db



def deep_L_layer_back_propagate(AL, Y, caches):
    
    L = len(caches)  # total number of layers
    Y = Y.reshape(AL.shape)
    grads = {}
    # initialize backpropagation
    dAL = -np.divide(Y, AL) + np.divide((1-Y), (1-AL))
    current_cache = caches[L-1]
    grads['dA'+str(L-1)], grads['dW'+str(L)], grads['db'+str(L)] = \
                     linear_activation_backward(dAL, current_cache, 'sigmoid')
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA'+str(l)], grads['dW'+str(l+1)], grads['db'+str(l+1)] = \
         linear_activation_backward(grads['dA'+str(l+1)], current_cache, 'relu')
            

    return grads
    




def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] = \
            parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = \
            parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters
    
    
    
    
    
    
    
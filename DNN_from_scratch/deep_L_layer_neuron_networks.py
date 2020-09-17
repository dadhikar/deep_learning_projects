"""Created on Mon Jun  8 13:30:48 2020 @author: dadhikar"""

import numpy  as np 
import helper_functions as hf
import gradient_checking as grad_check



def parameters_initialize(layers_dims):
    """
    

    Parameters
    ----------
    layers_dims : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    np.random.seed(5)
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1, L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],
                                                 layers_dims[l-1])*0.1
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))
        
        assert parameters['W'+str(l)].shape == (layers_dims[l],
                                                layers_dims[l-1])
        assert parameters['b'+str(l)].shape == (layers_dims[l], 1)
    return parameters




 
def one_layer_forward_propagation(A_prev, W, b, activation):
    """
    Parameters
    ----------
    A_prev : Matrix or Vector
        Activation matrix or vector from previous layer neuron
        of shape (n_nodes, n_samples)
    W : Matrix
        weight matrix of shape (n_layer, n_prev_layer)
    b : vector
        bias vector of shape (n_layer, 1)

    Returns
    -------
    None.

    """
    Z = np.dot(W, A_prev) + b
    
    assert Z.shape == (W.shape[0], A_prev.shape[1])
    
    if activation == 'sigmoid':
        A = hf.sigmoid(Z)
    elif activation == 'relu':
        A = hf.relu(Z)
    cache = (Z, A_prev, W, b)   
    return A, cache  
                       

def forward_propagation(X, parameters):
    """
    Parameters
    ----------
    X : Feature matrix or vector
        shape(n_fearures, n_samples)
    parameters : dictionary
        Ws and bs for neuron networks.

    Returns
    -------
    AL: activation of output layers
    caches = list of (Z, A_prev, W, b) for each layer

    """
    
    caches = []      # collecting Z, A, W,  and b for each layer in this list
    A = X
    L = len(parameters) // 2 
    
    for l in range(1, L):
        A_prev = A 
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = one_layer_forward_propagation(A_prev, W,
                                                 b, activation='relu')
        caches.append(cache)
    
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = one_layer_forward_propagation(A, W, b,  activation='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


def compute_cost(AL, Y):
    """
    

    Parameters
    ----------
    AL : Matrix or Vector
        activation of output layers
    Y : Matrix or Vector
       Target variable 

    Returns
    -------
    cost value

    """
    assert AL.shape == Y.shape
    m = Y.shape[1]
    
    cost1 = np.multiply(-Y, np.log(AL))
    cost2 = np.multiply(-(1-Y), (np.log(1-AL)))
    cost = (1/m)*np.sum(cost1 + cost2, keepdims=True)
    cost = np.squeeze(cost)
    return cost

def one_layer_backpropagation(dA, cache, activation):
    """
    Parameters
    ----------
    dA : TYPE
        DESCRIPTION.
    cache : TYPE
        DESCRIPTION.
    activation : TYPE
        DESCRIPTION.

    Returns
    -------
    dW : TYPE
        DESCRIPTION.
    db : TYPE
        DESCRIPTION.
    dA_prev : TYPE
        DESCRIPTION.

    """
    
    Z, A_prev, W, b = cache
    m = Z.shape[1]
    
    if activation == 'sigmoid':
        dZ = np.multiply(dA, hf.d_sigmoid(Z))
        
    elif activation == 'relu':
        dZ = np.multiply(dA, hf.d_relu(Z))
        
    assert dZ.shape == Z.shape   
        
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)* np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)    
        
    assert dW.shape == W.shape
    assert db.shape == b.shape
    assert dA_prev.shape == A_prev.shape
    
    return dW, db, dA_prev
    


def back_propagation(AL, Y, caches):
    """
    Parameters
    ----------
    AL : Matrix or Vector
        activation of output layers
    Y : Matrix or Vector
       Target variable
    caches : list 
        caches collected while forward propagation

    Returns
    -------
    grads : dictionary
        cost gradients with respect to 
        all the parameters w and b

    """
    
    
    grads = {}
    L =  len(caches)
    Y = Y.reshape(AL.shape)
    assert AL.shape == Y.shape
    
    dAL = -np.divide(Y, AL) + np.divide((1 - Y), (1 - AL))

    current_cache = caches[L-1]  # important step here
    grads['dW'+str(L)], grads['db'+str(L)], grads['dA'+str(L-1)] = \
        one_layer_backpropagation(dAL, current_cache, activation='sigmoid')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dW'+str(l+1)], grads['db'+str(l+1)], grads['dA'+str(l)] = \
        one_layer_backpropagation(grads['dA'+str(l+1)], current_cache,
                                  activation='relu')
        
    return grads 



def update_parameters(parameters, grads, learning_rate):
    """
    Parameters
    ----------
    parameters : dictionary 
        W and b parameters
    grads : dictionary
        dW and db gradients
    learning_rate : scalar

    Returns
    -------
    parameters : dictionary
        returns updated parameters

    """
    L = len(parameters) //2
    
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - \
                                             learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - \
                                             learning_rate*grads['db'+str(l+1)]
    return parameters                                         
        



def gradient_check_nn(parameters, gradients, X, Y, epsilon = 1e-7):
   
    
    theta_parameters = grad_check.dictionary_to_vector(parameters)
    theta_grad = grad_check.gradients_to_vector(gradients)
    # print(len(theta_grad))
    num_parameters = theta_parameters.shape[0]

    
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    for i in range(num_parameters):
        thetaplus = np.copy(theta_parameters)
        thetaplus[i][0] =  thetaplus[i][0]+ epsilon
        _parameters = grad_check.vector_to_dictionary(thetaplus,
                                                      parameters)
        AL, _ = forward_propagation(X, _parameters)                        
        J_plus[i]= compute_cost(AL, Y)
        
        
        thetaminus = np.copy(theta_parameters)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        _parameters1 = grad_check.vector_to_dictionary(thetaminus,
                                                       parameters)
        AL, _ = forward_propagation(X, _parameters1)                        
        J_minus[i] = compute_cost(AL, Y)
        
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    # print(len(J_minus))
    denominator = np.linalg.norm(theta_grad) + np.linalg.norm(gradapprox)              
    numerator = np.linalg.norm((theta_grad-gradapprox))                                   
    difference = numerator/denominator                                                  
    
    print(difference)
    
    return difference






""" Created on Fri Jun 12 11:56:15 2020 @author: dadhikar """

import numpy as np
import deep_L_layer_neuron_networks as dlnn


def dictionary_to_vector(parameters):
    """
    

    Parameters
    ----------
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    theta_vector : TYPE
        DESCRIPTION.
    parameters_shapes : TYPE
        DESCRIPTION.

    """
    
    count = 0
    for key in parameters.keys():
        
        # convert each parameters into a column vector 
        para_vector = parameters[key].reshape(-1,1)
        
        if count == 0:
            theta_vector = para_vector
        else:
           theta_vector = np.concatenate((theta_vector, para_vector), axis=0)
        
        count = count + 1

    return theta_vector

        
def vector_to_dictionary(theta, parameters):
    """
    

    Parameters
    ----------
    theta : TYPE
        DESCRIPTION.
    parameters : TYPE
        DESCRIPTION.

    Returns
    -------
    parameters_new : TYPE
        DESCRIPTION.

    """
    
    parameters_new = {}
    for key in parameters.keys():
        start = 0
        end = start + parameters[key].shape[0]*parameters[key].shape[1]
        row_size  = parameters[key].shape[0]
        column_size = parameters[key].shape[1]
        parameters_new[key] = theta[start:end].reshape((row_size, column_size))
        
        assert parameters_new[key].shape == parameters[key].shape
        
        start += end
        

    return parameters_new


def gradients_to_vector(gradients):
    """
    

    Parameters
    ----------
    gradients : TYPE
        DESCRIPTION.

    Returns
    -------
    theta_vector : TYPE
        DESCRIPTION.

    """
    
    
    count = 0
    for key in gradients.keys():
        if key.startswith('dA'):      # esaping dAs
            continue
        else:
            # convert each parameters into a column vector 
            grad_vector = gradients[key].reshape(-1,1)
        
            if count == 0:
               grad_theta_vector = grad_vector
            else:
               grad_theta_vector = np.concatenate((grad_theta_vector,
                                                   grad_vector), axis=0)
        
            count = count + 1

    return grad_theta_vector




def gradient_check_nn(parameters, gradients, X, Y, epsilon = 1e-7):
   
    # unrolling parameters into a vector
    theta_parameters = dictionary_to_vector(parameters)
    
    # unrolling gradients into a vector
    theta_grad = gradients_to_vector(gradients)
    
    
    num_parameters = theta_parameters.shape[0]

    
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    for i in range(num_parameters):
        thetaplus = np.copy(theta_parameters)
        thetaplus[i][0] =  thetaplus[i][0]+ epsilon
        _parameters = vector_to_dictionary(thetaplus, parameters)
        AL, _ = dlnn.forward_propagation(X, _parameters)                        
        J_plus[i]= dlnn.compute_cost(AL, Y)
        
        
        thetaminus = np.copy(theta_parameters)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        _parameters1 = dlnn.grad_check.vector_to_dictionary(thetaminus,
                                                       parameters)
        AL, _ = dlnn.forward_propagation(X, _parameters1)                        
        J_minus[i] = dlnn.compute_cost(AL, Y)
        
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    # print(len(J_minus))
    denominator = np.linalg.norm(theta_grad) + np.linalg.norm(gradapprox)              
    numerator = np.linalg.norm((theta_grad-gradapprox))                                   
    difference = numerator/denominator                                                  
    
    print(difference)
    
    return difference





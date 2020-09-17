""" Created on Thu Jun 11 18:25:41 2020 @author: dadhikar """

import numpy as np

def sigmoid(Z):
    """
    

    Parameters
    ----------
    Z : Matrix or Vector
        Net input matrix or vector

    Returns
    -------
    Activation matrix A

    """
    A = 1./(1 + np.exp(-Z))
    return A


def relu(Z):
    """
    Parameters
    ----------
    Z : Matrix or Vector
        Net input matrix or vector

    Returns
    -------
    Activation matrix A

    """
    A = np.maximum(0, Z)
    
    assert A.shape == Z.shape
    return A


def d_sigmoid(Z):
    """
    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    s = 1./ (1 + np.exp(-Z))
    
    dsigmoid = np.multiply(s, (1-s))
    
    return dsigmoid


def d_relu(Z):
    """
    

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # derivative of relu = 1 if z > 0, 0 if z<0
    drelu = np.where(Z<=0, 0, 1)
    
    assert drelu.shape == Z.shape
    
    return drelu


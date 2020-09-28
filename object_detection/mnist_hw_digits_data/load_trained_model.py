"""Created on Mon Sep 28 12:14:45 2020 @author: dadhikar"""
import os
from os.path import join
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


os.environ['KMP_DUPLICATE_LIB_OK']='True'

class dir_config:
    current_dir = os.getcwd()
    project_dir = "object_detection/mnist_hw_digits_data"
    model_dir = join(current_dir, project_dir, 'my_model')
    
    
    
def data_preprocessing():
    """
    

    Returns
    -------
    None.

    """
    # extracting training and testing datasets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # converting datatype and normalize training and testing data
    X_train = X_train.astype('float32')/X_train.max()
    X_test = X_test.astype('float32')/X_test.max()

    # one-hot vector representation of target level
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # split train and validation data
    X_train, X_valid = X_train[10000:], X_train[:10000]
    y_train, y_valid = y_train[10000:], y_train[:10000]
    return X_train, y_train, X_valid, y_valid, X_test, y_test    




# loading trained model
my_model = tf.keras.models.load_model(dir_config.model_dir + os.sep+ 
                                      'my_functional_model.h5')  

_, _, _, _, X_test, y_test = data_preprocessing()
predict_results = my_model.predict(X_test[0:64])
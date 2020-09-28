"""Created on Fri Sep 25 10:51:52 2020 @author: dadhikar"""

import os
import sys
from os.path import join

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model, to_categorical
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class dir_config:
    current_dir = os.getcwd()
    project_dir = "object_detection/mnist_hw_digits_data"
    model_dir = join(current_dir, project_dir, 'my_model')
    
    
    
class model_hype_para:
     epochs = 100;
     batch_size = 256;
     
# sys.exit() 


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

# controlling training state
# train_state = input('Enter True or False: ')   

def create_functional_model(train_model=True, epochs=model_hype_para.epochs,
                            batch_size=model_hype_para.batch_size):
    """
    functional approach of model generation

    Returns
    -------
    model : TYPE
        DESCRIPTION.
    summary : TYPE
        DESCRIPTION.
    model_plot : TYPE
        DESCRIPTION.

    """
    if train_model==False:
        print('No model training!!')
    else:
        input_ = tf.keras.layers.Input(shape=(28, 28))
        flatten_input = tf.keras.layers.Flatten(input_shape=(28, 28))(input_)
        hidden_l1 = tf.keras.layers.Dense(256, activation='relu')(flatten_input)
        # drop_out = tf.keras.layers.Dropout(rate=0.4)(hidden_l1)
        hidden_l2 = tf.keras.layers.Dense(128, activation='relu')(hidden_l1)
        # hidden_l3 = tf.keras.layers.Dense(128, activation='relu')(hidden_l2)
        # hidden_l4 = tf.keras.layers.Dense(64, activation='relu')(hidden_l3)
        output = tf.keras.layers.Dense(10, activation='softmax')(hidden_l2)
        # create a model
        model = tf.keras.Model(inputs=input_, outputs=output)
        # plot and save the model just created
        plot_model(model, to_file=dir_config.model_dir + os.sep +'functional_model_plot.png',
                   show_shapes=True, show_layer_names=True)
        
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01, rho=0.95,
                                             epsilon=1e-07,name='Adadelta')
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                   label_smoothing=0,
                                                   name='categorical_crossentropy')
        
        # configure the model for training using comiple() method
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        # calling data 
        X_train, y_train, X_valid, y_valid, _, _ = data_preprocessing()
        
        fit_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(X_valid, y_valid))
        
        #saving trained model
        model.save(dir_config.model_dir + os.sep+ 'my_functional_model.h5')
    
    return model, fit_history



def create_sequential_model(train_model=True, epochs=model_hype_para.epochs,
                            batch_size=model_hype_para.batch_size):
    
    """
    create a sequential model

    Parameters
    ----------
     : TYPE
     DESCRIPTION.

    Returns
    -------
    None.

    """
    if train_model==False:
        print('No model training!!')
    else:
        model = tf.keras.Sequential()  # instantiate a sequential model
    
        # adding different layers to a model
        #Flattens the input. Does not affect the batch size.
        model.add(tf.keras.layers.Flatten(input_shape=[28, 28])) 
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        # model.add(tf.keras.layers.Dropout(rate=0.45))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
         
        # plot and save the model just created
        plot_model(model, to_file= dir_config.model_dir + os.sep +'sequential_model_plot.png',
                   show_shapes=True, show_layer_names=True)
         
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.01, rho=0.90,
                                             epsilon=1e-07,name='Adadelta')
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                   label_smoothing=0,
                                                   name='categorical_crossentropy')
        # configure the model for training using comiple() method
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        
        # calling data 
        X_train, y_train, X_valid, y_valid, _, _ = data_preprocessing()
        
        fit_history = model.fit(X_train, y_train, epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_valid, y_valid))
        # saving model
        model.save(dir_config.model_dir + os.sep+ 'my_sequential_model.h5')
    
    return model, fit_history



# select which model want to train
my_model, fit_history = create_sequential_model(train_model=True)
# my_model, fit_history = create_functional_model(train_model=True)

def model_details(model=my_model):
    """
    

    Returns
    -------
    None.

    """
    print('The model summary: \n')
    print(model.summary())


model_details()




def plot_train_results(history = fit_history):
    train_loss = history.history['loss']
    train_accuracy = history.history['accuracy']
    val_loss = history.history['val_loss'] 
    val_accuracy = history.history['val_accuracy']
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
    ax[0].plot(range(model_hype_para.epochs), np.asarray(train_loss),
               'ro',label='Training loss')
    ax[0].plot(range(model_hype_para.epochs), np.asarray(val_loss),
               'go',label='Validation loss')
    ax[0].set_xlabel('No. of epochs')
    ax[0].set_ylabel('Loss (arb. unit)')
    ax[0].legend()
    ax[1].plot(range(model_hype_para.epochs), np.asarray(train_accuracy)*100,
               'bo', label='Training accuracy')
    ax[1].plot(range(model_hype_para.epochs), np.asarray(val_accuracy)*100,
               'ko',label='Validation accuracy')
    ax[1].set_xlabel('No. of epochs')
    ax[1].set_ylabel('Accuracy(%)')
    ax[1].legend()



# plotting training results
plot_train_results()





def test_results(y_target_pos):
    
    # calling data 
    _, _, _, _, X_test, y_test = data_preprocessing()
    
    # plotting first digit in the training dataset
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.imshow(X_test[y_target_pos], cmap='gray', alpha=0.8)
    plt.axis('off')
    plt.show()
    test_loss, test_accuracy = my_model.evaluate(x=X_test, y=y_test,
                                              batch_size=64)
    
    print('The first digit in the training dataset is: ', y_test[y_target_pos])
    print('Test loss: ', test_loss, ',',  'Test accuracy: ', test_accuracy)
    
    
test_results(int(input('Enter the test data position: '))) 

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D 

"""
A File for cnn model construction based on hyperparameters
"""

def build_model(params, target_size):
    '''
    Function to build cnn model

    :param params : hyperparameters to build cnn
    :param target_size : size of the final softmax layer

    :returns model object
    '''

    filters = params.filters
    kernel_size = params.kernel_size
    padding = params.padding
    strides = params.strides
    activation = params.activation
    dense = params.dense
    dropout1 = params.dropout1
    dropout2 = params.dropout2
    pool_size = params.pool_size

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(9, 64, 1)))

    for i in range(len(filters)):
        if i != 0:
            model.add(Conv2D(filters = filters[i], 
                            kernel_size = kernel_size, 
                            padding = padding, 
                            strides = strides,
                            activation = activation))
        model.add(Conv2D(filters = filters[i], 
                        kernel_size = kernel_size, 
                        padding = padding, 
                        strides = strides,
                        activation = activation))
        model.add(MaxPool2D(pool_size = pool_size, 
                            padding = 'same'))
        if (dropout1 != None):
            model.add(Dropout(dropout1))

    model.add(Flatten())

    for i in range(len(dense)):
        model.add(Dense(dense[i], activation = activation))

    if (dropout2 != None):
        model.add(Dropout(dropout2))
    
    model.add(Dense(target_size, activation = "softmax"))

    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy'],
        metrics=['accuracy']
    )
    print(model.layers)
    print(len(model.layers))
    model.summary()
    
    return model

def build_mo_body(params):

    filters = params.filters
    kernel_size = params.kernel_size
    padding = params.padding
    strides = params.strides
    activation = params.activation
    dense = params.dense
    dropout1 = params.dropout1
    dropout2 = params.dropout2
    pool_size = params.pool_size

    inputs = tf.keras.Input(shape=(9, 64, 1))
    x = Conv2D(filters = filters[0], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(inputs)
    x = MaxPool2D(pool_size = pool_size)(x)
    
    x = Conv2D(filters = filters[1], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(x)
    x = Conv2D(filters = filters[1], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(x)
    x = MaxPool2D(pool_size = pool_size, padding = 'same')(x)
    
    x = Conv2D(filters = filters[2], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(x)
    x = Conv2D(filters = filters[2], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(x)
    x = MaxPool2D(pool_size = pool_size, padding = 'same')(x)
    
    x = Conv2D(filters = filters[3], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(x)
    x = Conv2D(filters = filters[3], kernel_size = kernel_size, padding = 'same', strides = strides,activation = 'relu')(x)
    x = MaxPool2D(pool_size = pool_size, padding = 'same')(x)
    
    x = Flatten()(x)

    x = Dense(dense[0], activation = "relu")(x)
    x = Dense(dense[1], activation = "relu")(x)
    x = Dense(dense[2], activation = "relu")(x)
    if (dropout2 != None):
        x = Dropout(dropout2)(x)

    return inputs, x


def build_mo_head(name, x, target_size):
    head = Dense(target_size, activation = "softmax", name=name)(x)
    return head

# if __name__ == "__main__":
    # build_model(filters=(32,64,128,256),
    #             kernel_size=(3,3),
    #             padding='same',
    #             strides=1,
    #             activation='relu',
    #             pool_size=(2,2),
    #             dropout1=0.2,
    #             dropout2=0.2,
    #             target_size=5)

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D # Lambda, 
# from tensorflow.keras.layers import Reshape # , Conv2DTranspose
# from tensorflow.keras.models import Model
# from tensorflow.keras.losses import mse, binary_crossentropy
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.callbacks import CSVLogger
# from tensorflow.keras import backend as K


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(9, 64, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (9,3), padding = 'same', strides = 1,activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters = 64, kernel_size = (9,3), padding = 'same', strides = 1,activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    # model.add(Conv2D(filters = 128, kernel_size = (9,3), padding = 'same', strides = 1,activation = 'relu'))
    # model.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(filters = 128, kernel_size = (9,3), padding = 'same', strides = 1,activation = 'relu'))
    # model.add(MaxPool2D(pool_size = (2, 2), padding = 'same'))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu'))
    # model.add(MaxPool2D(pool_size = (3,3), padding = 'same'))
    # model.add(Dropout(0.2))


    # model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu'))
    # model.add(MaxPool2D(pool_size = (3,3)))
    # model.add(Dropout(0.2))


    model.add(Flatten())
    model.add(Dense(16, activation = "relu"))

    model.add(Dropout(0.2))
    model.add(Dense(5, activation = "softmax"))

    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy'],
        metrics=['accuracy']
    )
    model.summary()
    return model
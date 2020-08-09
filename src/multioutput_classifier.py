import argparse
import numpy as np
import ast

from tensorflow.keras.utils import to_categorical

from constants import GET_DEFAULTS, TARGET_GENRES, TARGET_SECTIONS
from utilities import load_data, split_data, write_csv
from plot_data import plot_confusion_matrix, plot_epochs, plot_epochs_MO
from evaluation import confusion_mtx, f_score

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D
from collections import Counter
from sklearn.model_selection  import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model

def build_model_1(params, inputs, target_size, name):
    filters = params.filters
    kernel_size = params.kernel_size
    padding = params.padding
    strides = params.strides
    activation = params.activation
    dense = params.dense
    dropout1 = params.dropout1
    dropout2 = params.dropout2
    pool_size = params.pool_size

    x = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(inputs)
    x = MaxPool2D(pool_size = (2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(x)
    x = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(x)
    x = MaxPool2D(pool_size = (2, 2), padding = 'same')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(x)
    x = Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(x)
    x = MaxPool2D(pool_size = (2, 2), padding = 'same')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(x)
    x = Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', strides = 1,activation = 'relu')(x)
    x = MaxPool2D(pool_size = (2, 2), padding = 'same')(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)

    x = Dense(256, activation = "relu")(x)
    x = Dense(128, activation = "relu")(x)
    x = Dense(32, activation = "relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(target_size, activation = "softmax", name=name)(x)

    return x

def build_model(params, target_size):

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
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', 
        type = str, 
        default = GET_DEFAULTS["model_path"],
        help = 'path to model file, default ' + GET_DEFAULTS["model_path"]
    )
    parser.add_argument(
        '--Xdata', 
        type = str, 
        default = GET_DEFAULTS["X_both"],
        help = 'path to input data file, default ' + GET_DEFAULTS["X_both"]
    )
    parser.add_argument(
        '--ydata1', 
        type = str, 
        default = GET_DEFAULTS["y1_both"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y1_both"]
    )
    parser.add_argument(
        '--ydata2', 
        type = str, 
        default = GET_DEFAULTS["y2_both"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y2_both"]
    )
    parser.add_argument(
        '--filters', 
        type = ast.literal_eval, 
        default = GET_DEFAULTS["filters"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["filters"])
    )
    parser.add_argument(
        '--kernel_size', 
        type = ast.literal_eval, 
        default = GET_DEFAULTS["kernel_size"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["kernel_size"])
    )
    parser.add_argument(
        '--padding', 
        type = str, 
        default = GET_DEFAULTS["padding"],
        help = 'path to target data file, default ' + GET_DEFAULTS["padding"]
    )
    parser.add_argument(
        '--strides', 
        type = int, 
        default = GET_DEFAULTS["strides"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["strides"])
    )
    parser.add_argument(
        '--activation', 
        type = str, 
        default = GET_DEFAULTS["activation"],
        help = 'path to target data file, default ' + GET_DEFAULTS["activation"]
    )
    parser.add_argument(
        '--pool_size', 
        type = ast.literal_eval, 
        default = GET_DEFAULTS["pool_size"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["pool_size"])
    )
    parser.add_argument(
        '--dense', 
        type = tuple, 
        default = GET_DEFAULTS["dense"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["dense"])
    )
    parser.add_argument(
        '--dropout1', 
        type = str, 
        default = GET_DEFAULTS["dropout1"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["dropout1"])
    )
    
    parser.add_argument(
        '--dropout2', 
        type = float, 
        default = GET_DEFAULTS["dropout2"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["dropout2"])
    )

    parser.add_argument(
        '--modelname', 
        type = str, 
        help = 'path to target data file, default '
    )

    params = parser.parse_args()
    # open("params.txt", "w").write(str(params))
    filters = params.filters
    kernel_size = params.kernel_size
    padding = params.padding
    strides = params.strides
    activation = params.activation
    dense = params.dense
    dropout1 = params.dropout1
    dropout2 = params.dropout2
    pool_size = params.pool_size
    
    X, y1_, y2_ = load_data(params.Xdata, params.ydata1, params.ydata2)
    print(X.shape, y1_.shape, y2_.shape)
    
    y1 = to_categorical(y1_)
    y2 = to_categorical(y2_)
    
    print(Counter(y1_))
    print(Counter(y2_))

    split = train_test_split(X, y1, y2,	test_size=0.1, random_state=42)
    (trainX, testX, trainGenreY, testGenreY, trainSectionY, testSectionY) = split

    print(trainX.shape, testX.shape, trainGenreY.shape, testGenreY.shape, trainSectionY.shape, testSectionY.shape)
    trainX = np.reshape(trainX, [-1, 9, 64, 1])
    testX = np.reshape(testX, [-1, 9, 64, 1])

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

    # x = Dense(target_size, activation = "softmax", name=name)(x)
    genre_model = Dense(5, activation = "softmax", name="genre_output")(x)
    section_model = Dense(5, activation = "softmax", name="section_output")(x)

    model = Model(inputs=inputs, outputs=[genre_model, section_model], name="main_model")
    model.summary()
    losses = {
        "genre_output": "categorical_crossentropy",
        "section_output": "categorical_crossentropy",
    }
    lossWeights = {"genre_output": 1.0, "section_output": 1.0}

    from tensorflow.keras.optimizers import Adam
    opt = Adam(lr=0.001, decay=0.001 / 40)
    # model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    model.compile(
        optimizer='adam',
        loss=losses, loss_weights=lossWeights, metrics=["accuracy"])
    history = model.fit(x=trainX, y={"genre_output": trainGenreY, "section_output": trainSectionY},
                        validation_data=(testX,	{"genre_output": testGenreY, "section_output": testSectionY}),
                        epochs=40, verbose=1)
    
    (y_pred1, y_pred2) = model.predict(testX)
    y_conf_matrix1 = confusion_mtx(testGenreY, y_pred1)
    y_conf_matrix2 = confusion_mtx(testSectionY, y_pred2)
    f1_score1, cls_report1 = f_score(testGenreY, y_pred1)
    f1_score2, cls_report2 = f_score(testSectionY, y_pred2)
    
    
    model.save(GET_DEFAULTS['model_file_MO'] + params.modelname + '.h5')
    plot_epochs_MO(history, 40, GET_DEFAULTS['model_file_MO'] + params.modelname)
      
    result = []
    result.append(params.modelname)
    result.append(history.history['loss'][-1])
    result.append(history.history['val_genre_output_accuracy'][-1])
    result.append(history.history['val_section_output_accuracy'][-1])
    result.append(history.history['val_genre_output_loss'][-1])
    result.append(history.history['val_section_output_loss'][-1])
    result.append(f1_score1)
    result.append(f1_score2)
    result.append("\"" + str(y_conf_matrix1) + "\"")
    result.append("\"" + str(y_conf_matrix2) + "\"")
    result.append("\"" + str(cls_report1) + "\"")
    result.append("\"" + str(cls_report2) + "\"")

    write_csv(result, 'result_MO')

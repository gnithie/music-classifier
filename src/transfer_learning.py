import argparse
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, Dropout

from constants import GET_DEFAULTS, TARGET_SECTIONS
from utilities import load_data, split_data
from cnn_model import build_model
from plot_data import plot_confusion_matrix

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
        default = GET_DEFAULTS["X_data1"],
        help = 'path to input data file, default ' + GET_DEFAULTS["X_data1"]
    )
    parser.add_argument(
        '--ydata', 
        type = str, 
        default = GET_DEFAULTS["y_data1"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y_data1"]
    )
    params = parser.parse_args()

    X, y = load_data(params.Xdata, params.ydata)
    y = to_categorical(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_train.shape[1])
    base_model = load_model(GET_DEFAULTS['model_file'])
    
    model = tf.keras.Sequential(base_model.layers[:-1])
    for layer in model.layers:
	    layer.trainable = False

    model.add(Dense(y_train.shape[1], activation='softmax', name='output'))
    model.summary()
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy'],
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test) 

    y_conf_matrix = confusion_matrix(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1))
    cmn = np.around(y_conf_matrix.astype('float') / y_conf_matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
    print(cmn)
    # plot_confusion_matrix(cmn, TARGET_SECTIONS)
    f1_score = precision_recall_fscore_support(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), average='macro')
    print(f1_score)

    cls_report = classification_report(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), digits=4)
    print(cls_report)
import argparse
import numpy as np

from sklearn.metrics import confusion_matrix

from tensorflow.keras.utils import to_categorical

from constants import GET_DEFAULTS, TARGET_GENRES
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
        default = GET_DEFAULTS["X_data"],
        help = 'path to input data file, default ' + GET_DEFAULTS["X_data"]
    )
    parser.add_argument(
        '--ydata', 
        type = str, 
        default = GET_DEFAULTS["y_data"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y_data"]
    )
    params = parser.parse_args()

    X, y = load_data(params.Xdata, params.ydata)
    y = to_categorical(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model()
    
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)
    model.save_weights('1.h5')

    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test) 

    y_conf_matrix = confusion_matrix(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1))
    cmn = np.around(y_conf_matrix.astype('float') / y_conf_matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
    print(cmn)
    plot_confusion_matrix(cmn, TARGET_GENRES)

    
import argparse
import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

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
        default = GET_DEFAULTS["X_data1"],
        help = 'path to input data file, default ' + GET_DEFAULTS["X"]
    )
    parser.add_argument(
        '--ydata', 
        type = str, 
        default = GET_DEFAULTS["y_data1"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y"]
    )
    parser.add_argument(
        '--filters', 
        type = tuple, 
        default = GET_DEFAULTS["filters"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["filters"])
    )
    parser.add_argument(
        '--kernel_size', 
        type = tuple, 
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
        type = tuple, 
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
        type = str, 
        default = GET_DEFAULTS["dropout2"],
        help = 'path to target data file, default ' + str(GET_DEFAULTS["dropout2"])
    )
    params = parser.parse_args()
    
    X, y = load_data(params.Xdata, params.ydata)
    y = to_categorical(y)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model(params, y_train.shape[1])
    
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

    model.save(GET_DEFAULTS['model_file'])

    model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test) 

    y_conf_matrix = confusion_matrix(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1))
    cmn = np.around(y_conf_matrix.astype('float') / y_conf_matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
    print(cmn)
    # plot_confusion_matrix(cmn, TARGET_GENRES)

    f1_score = precision_recall_fscore_support(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), average='macro')
    print(f1_score)

    cls_report = classification_report(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), digits=4)
    print(cls_report)
    
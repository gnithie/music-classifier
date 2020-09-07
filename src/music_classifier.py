import argparse
import numpy as np
import ast

from tensorflow.keras.utils import to_categorical

from constants import GET_DEFAULTS, TARGET_GENRES
from utilities import load_data, split_data, write_csv
from cnn_model import build_model
from plot_data import plot_confusion_matrix, plot_epochs
from evaluation import confusion_mtx, f_score

"""
A file to construct Base Model for Music classification 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', 
        type = str, 
        default = GET_DEFAULTS["model_path"],
        help = 'path to save model file, default ' + GET_DEFAULTS["model_path"]
    )
    parser.add_argument(
        '--Xdata', 
        type = str, 
        default = GET_DEFAULTS["X"],
        help = 'path to input data file, default ' + GET_DEFAULTS["X"]
    )
    parser.add_argument(
        '--ydata', 
        type = str, 
        default = GET_DEFAULTS["y"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y"]
    )
    parser.add_argument(
        '--filters', 
        type = ast.literal_eval, 
        default = GET_DEFAULTS["filters"],
        help = 'CNN filters, default ' + str(GET_DEFAULTS["filters"])
    )
    parser.add_argument(
        '--kernel_size', 
        type = ast.literal_eval, 
        default = GET_DEFAULTS["kernel_size"],
        help = 'CNN kernel size, default ' + str(GET_DEFAULTS["kernel_size"])
    )
    parser.add_argument(
        '--padding', 
        type = str, 
        default = GET_DEFAULTS["padding"],
        help = 'CNN padding, default ' + GET_DEFAULTS["padding"]
    )
    parser.add_argument(
        '--strides', 
        type = int, 
        default = GET_DEFAULTS["strides"],
        help = 'CNN strides, default ' + str(GET_DEFAULTS["strides"])
    )
    parser.add_argument(
        '--activation', 
        type = str, 
        default = GET_DEFAULTS["activation"],
        help = 'CNN activation, default ' + GET_DEFAULTS["activation"]
    )
    parser.add_argument(
        '--pool_size', 
        type = ast.literal_eval, 
        default = GET_DEFAULTS["pool_size"],
        help = 'CNN maxpool size, default ' + str(GET_DEFAULTS["pool_size"])
    )
    parser.add_argument(
        '--dense', 
        type = tuple, 
        default = GET_DEFAULTS["dense"],
        help = 'CNN dense layer size list' + str(GET_DEFAULTS["dense"])
    )
    parser.add_argument(
        '--dropout1', 
        type = str, 
        default = GET_DEFAULTS["dropout1"],
        help = 'Dropout in CNN layer, default ' + str(GET_DEFAULTS["dropout1"])
    )
    
    parser.add_argument(
        '--dropout2', 
        type = float, 
        default = GET_DEFAULTS["dropout2"],
        help = 'Dropout before softmax, default ' + str(GET_DEFAULTS["dropout2"])
    )

    parser.add_argument(
        '--modelname', 
        type = str, 
        help = 'model filename'
    )

    params = parser.parse_args()
    
    # load data from npy files 
    X, y = load_data(params.Xdata, params.ydata)
    y = to_categorical(y)
    print(X.shape, y.shape)

    # split data for Train and Test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # build CNN model
    model = build_model(params, y_train.shape[1])
    
    # fit the CNN model. 20% of data is set for validation during training
    history = model.fit(X_train, y_train, epochs=40, batch_size=128, validation_split=0.2)

    # Model evluation
    test_history = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test) 

    # model evaluation. Confustion Matrix and f score
    y_conf_matrix = confusion_mtx(y_test, y_pred)
    f1_score, cls_report = f_score(y_test, y_pred)

    # save model
    model.save(GET_DEFAULTS['model_file_G'] + params.modelname+ '.h5')

    # plot learning curve for loss and accuracy
    plot_epochs(history, 40, GET_DEFAULTS['model_file_G'] + params.modelname)
        
    # log metrics into csv file
    result = []
    result.append(history.history['accuracy'][-1])
    result.append(history.history['loss'][-1])
    result.append(history.history['val_accuracy'][-1])
    result.append(history.history['val_loss'][-1])
    result.append(test_history[1])
    result.append(test_history[0])
    result.append(f1_score)
    result.append("\"" + str(y_conf_matrix) + "\"")
    result.append("\"" + str(cls_report) + "\"")
    write_csv(result, 'result_BM')

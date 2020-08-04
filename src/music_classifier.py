import argparse
import numpy as np
import ast

from tensorflow.keras.utils import to_categorical

from constants import GET_DEFAULTS, TARGET_GENRES
from utilities import load_data, split_data, write_csv
from cnn_model import build_model
from plot_data import plot_confusion_matrix, plot_epochs
from evaluation import confusion_matrix, f_score


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
    open("params.txt", "w").write(str(params))
    
    X, y = load_data(params.Xdata, params.ydata)
    y = to_categorical(y)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_model(params, y_train.shape[1])
    
    history = model.fit(X_train, y_train, epochs=40, batch_size=128, validation_split=0.2)

    test_history = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test) 

    y_conf_matrix = confusion_matrix(y_test, y_pred)
    f1_score, cls_report = f_score(y_test, y_pred)

    
    model.save(GET_DEFAULTS['model_file_G'] + params.modelname+ '.h5')
    plot_epochs(history, 40, GET_DEFAULTS['model_file_G'] + params.modelname)
    # plot_confusion_matrix(cmn, TARGET_GENRES)
    
    # y_conf_matrix = confusion_matrix(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1))
    # cmn = np.around(y_conf_matrix.astype('float') / y_conf_matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
    # cmn
    # print(cmn)
    # # plot_confusion_matrix(cmn, TARGET_GENRES)

    # f1_score = precision_recall_fscore_support(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), average='macro')
    # f1_score
    # print(f1_score)

    # cls_report = classification_report(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), digits=4)
    # cls_report
    # print(cls_report)
    
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
    # res = ','.join(str(v) for v in result)
    # f= open("../output/result_S_result.csv","a+")
    # f.write(res + '\n')
    # f.close()

import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization, Dropout

from constants import GET_DEFAULTS, TARGET_SECTIONS
from utilities import load_data, split_data, write_csv
from cnn_model import build_model
from plot_data import plot_confusion_matrix, plot_epochs
from evaluation import confusion_mtx, f_score

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
        default = GET_DEFAULTS["X1"],
        help = 'path to input data file, default ' + GET_DEFAULTS["X1"]
    )
    parser.add_argument(
        '--ydata', 
        type = str, 
        default = GET_DEFAULTS["y1"],
        help = 'path to target data file, default ' + GET_DEFAULTS["y1"]
    )
    parser.add_argument(
        '--modelname', 
        type = str, 
        help = 'path to target data file, default '
    )
    parser.add_argument(
        '--embedding_val', 
        type = int, 
        help = 'path to target data file, default '
    )
    parser.add_argument(
        '--train_layer', 
        type = int, 
        default = 0,
        help = 'path to target data file, default '
    )
    params = parser.parse_args()

    X, y = load_data(params.Xdata, params.ydata)
    y = to_categorical(y)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    base_model = load_model(GET_DEFAULTS['model_file_G'] + params.modelname+ '.h5')
    
    model = tf.keras.Sequential(base_model.layers[:params.embedding_val])
    if params.train_layer == 0:
        for layer in model.layers:
            layer.trainable = False
    else:
        for layer in range(0, params.train_layer):
            model.layers[layer].trainable = False
        for layer in range(params.train_layer, len(model.layers)):
            model.layers[layer].trainable = True

    model.add(Dense(y_train.shape[1], activation='softmax', name='output'))
    model.summary()
    model.compile(
        optimizer='adam',
        loss=['categorical_crossentropy'],
        metrics=['accuracy']
    )
    history = model.fit(X_train, y_train, epochs=40, batch_size=128, validation_split=0.2)
    
    test_history = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test) 

    y_conf_matrix = confusion_mtx(y_test, y_pred)
    f1_score, cls_report = f_score(y_test, y_pred)

    # y_conf_matrix = confusion_matrix(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1))
    # cmn = np.around(y_conf_matrix.astype('float') / y_conf_matrix.sum(axis=1)[:, np.newaxis], decimals = 2)
    # print(cmn)
    # # plot_confusion_matrix(cmn, TARGET_SECTIONS)
    # f1_score = precision_recall_fscore_support(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), average='macro')
    # print(f1_score)

    # cls_report = classification_report(np.argmax(y_test, axis =1), np.argmax(y_pred, axis =1), digits=4)
    # print(cls_report)

    model.save(GET_DEFAULTS['model_file_TL_S'] + params.modelname+ '_' + str(params.train_layer) + '.h5')
    plot_epochs(history, 40, GET_DEFAULTS['model_file_TL_S'] + params.modelname+ '_' + str(params.train_layer))

    result = []
    result.append(params.modelname)
    result.append(history.history['accuracy'][-1])
    result.append(history.history['loss'][-1])
    result.append(history.history['val_accuracy'][-1])
    result.append(history.history['val_loss'][-1])
    result.append(test_history[1])
    result.append(test_history[0])
    result.append(f1_score)
    result.append("\"" + str(y_conf_matrix) + "\"")
    result.append("\"" + str(cls_report) + "\"")
    write_csv(result, 'result_TL')
    # res = ','.join(str(v) for v in result)
    # f= open("../output/result_TL_G_result.csv","a+")
    # f.write(res + '\n')
    # f.close()
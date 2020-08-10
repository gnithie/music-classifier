import argparse
import numpy as np
import ast
from collections import Counter

from sklearn.model_selection  import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

from constants import GET_DEFAULTS, TARGET_GENRES, TARGET_SECTIONS
from utilities import load_data, split_data, write_csv
from plot_data import plot_confusion_matrix, plot_epochs, plot_epochs_MO
from evaluation import confusion_mtx, f_score
from cnn_model import build_mo_body, build_mo_head

"""
A file to construct Multi-Output(Multi-Head) classification
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    
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
        help = 'path to first target data file, default ' + GET_DEFAULTS["y1_both"]
    )
    parser.add_argument(
        '--ydata2', 
        type = str, 
        default = GET_DEFAULTS["y2_both"],
        help = 'path to second target data file, default ' + GET_DEFAULTS["y2_both"]
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
        help = 'CNN dense layer size list, default ' + str(GET_DEFAULTS["dense"])
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
    X, y1_, y2_ = load_data(params.Xdata, params.ydata1, params.ydata2)
    print(X.shape, y1_.shape, y2_.shape)
    
    y1 = to_categorical(y1_)
    y2 = to_categorical(y2_)
    
    print(Counter(y1_))
    print(Counter(y2_))

    # split data for train and test
    split = train_test_split(X, y1, y2,	test_size=0.1, random_state=42)
    (X_train, X_test, y_genre_train, y_genre_test, y_section_train, y_section_test) = split

    # reshape data
    print(X_train.shape, X_test.shape, y_genre_train.shape, y_genre_test.shape, y_section_train.shape, y_section_test.shape)
    X_train = np.reshape(X_train, [-1, 9, 64, 1])
    X_test = np.reshape(X_test, [-1, 9, 64, 1])

    # build multi output body layers. All CNN layers are added except final softmax layer
    inputs, x = build_mo_body(params)

    # build multi output head layers. Softmax layers only considered as head. One head for Genre labels and another head for Section labels.
    genre_model = build_mo_head('genre_output', x, y_genre_train.shape[1])
    section_model = build_mo_head('section_output', x, y_section_train.shape[1])

    # construct the model with multi output body adn head details
    model = Model(inputs=inputs, outputs=[genre_model, section_model], name="main_model")
    model.summary()

    # dictionary to contain losses for genre and section labels
    losses = {
        "genre_output": "categorical_crossentropy",
        "section_output": "categorical_crossentropy",
    }
    lossWeights = {"genre_output": 1.0, "section_output": 1.0}

    model.compile(
        optimizer='adam',
        loss=losses, loss_weights=lossWeights, metrics=["accuracy"])

    # fit the model. 20% of data is set for validation during training.
    history = model.fit(x=X_train, y={"genre_output": y_genre_train, "section_output": y_section_train},
                        validation_data=(X_test,	{"genre_output": y_genre_test, "section_output": y_section_test}),
                        epochs=40, verbose=1)
    
    # model evaluation
    (y_genre_pred, y_section_pred) = model.predict(X_test)

    # model evaluation metrics
    y_genre_conf_matrix = confusion_mtx(y_genre_test, y_genre_pred)
    y_section_conf_matrix = confusion_mtx(y_section_test, y_section_pred)
    fscore_genre, cls_report_genre = f_score(y_genre_test, y_genre_pred)
    fscore_section, cls_report_section = f_score(y_section_test, y_section_pred)
    
    # save the model 
    model.save(GET_DEFAULTS['model_file_MO'] + params.modelname + '.h5')

    # plot the learning curve
    plot_epochs_MO(history, 40, GET_DEFAULTS['model_file_MO'] + params.modelname)
      
    # log metrics into csv file
    result = []
    result.append(params.modelname)
    result.append(history.history['loss'][-1])
    result.append(history.history['val_genre_output_accuracy'][-1])
    result.append(history.history['val_section_output_accuracy'][-1])
    result.append(history.history['val_genre_output_loss'][-1])
    result.append(history.history['val_section_output_loss'][-1])
    result.append(fscore_genre)
    result.append(fscore_section)
    result.append("\"" + str(y_genre_conf_matrix) + "\"")
    result.append("\"" + str(y_section_conf_matrix) + "\"")
    result.append("\"" + str(cls_report_genre) + "\"")
    result.append("\"" + str(cls_report_section) + "\"")

    write_csv(result, 'result_MO')

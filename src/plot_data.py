import numpy as np
import os

import seaborn as sb
import matplotlib.pyplot as plt
import umap
import umap.plot

from tensorflow.keras.models import load_model, Model
import tensorflow as tf

from constants import GET_DEFAULTS, TARGET_GENRES, TARGET_SECTIONS
from utilities import load_data, split_data

"""
A file to plot data
"""

def plot_confusion_matrix(data, label, filename='conf_mtx_plot'):
    '''
    Function to plot confusion matrix

    :param data : confusion matrix resultant matrix
    :param label : target label values for matrix
    :filename : filename for confusion matrix plot. By default, filename is conf_mtx_plot.
    '''

    sb.heatmap(data, annot = True, xticklabels = label, yticklabels = label, cmap="YlGnBu", fmt='g')
    plt.savefig(filename + '_cm.png')
    plt.close()

def plot_epochs(history, epochs, filename='learning_curve_plot'):
    '''
    Function to plot learning curve

    :param : model fit result 
    :param : epochs passed for fit function
    :filename : filename for learning curve. By default, filename is learning_curve_plot.
    '''

    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
    ax1, ax2 = axes
    ax1.plot(range(epochs), history.history['accuracy'], label = 'Training Accuracy')
    ax1.plot(range(epochs), history.history['val_accuracy'], label = 'Validation Accuracy')
    ax1.set_ylim(0, 1)
    ax2.plot(range(epochs), history.history['loss'], label = 'Training Loss')
    ax2.plot(range(epochs), history.history['val_loss'], label = 'Validation Loss')
    ax1.set_title('Traiing and Validation Accuracy')
    ax2.set_title('Training and Validation Loss')
    ax1.set_xlabel('Number of Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Number of Epochs')
    ax1.legend()
    ax2.legend()
    plt.savefig(filename + '.png')
    plt.close()

def plot_epochs_MO(history, epochs, filename):
    '''
    Function to plot learning curve for Multi Output model

    :param : model fit result 
    :param : epochs passed for fit function
    :filename : filename for learning curve. By default, filename is learning_curve_plot.
    '''

    plt.style.use("ggplot")
    lossNames = ["loss", "genre_output_loss", "section_output_loss"]
    (fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
    # loop over the loss names
    for (i, l) in enumerate(lossNames):
        # plot the loss for both the training and validation data
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(np.arange(0, 40), history.history[l], label=l)
        ax[i].plot(np.arange(0, 40), history.history["val_" + l],
            label="val_" + l)
        ax[i].legend()
    # save the losses figure
    plt.tight_layout()
    plt.savefig(filename + '_losses.png')
    plt.close()
    accuracyNames = ["genre_output_accuracy", "section_output_accuracy"]
    (fig, ax) = plt.subplots(2, 1, figsize=(8, 8))
    # loop over the accuracy names
    for (i, l) in enumerate(accuracyNames):
        # plot the loss for both the training and validation data
        ax[i].set_title("Accuracy for {}".format(l))
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Accuracy")
        ax[i].plot(np.arange(0, 40), history.history[l], label=l)
        ax[i].plot(np.arange(0, 40), history.history["val_" + l],
            label="val_" + l)
        ax[i].legend()
    # save the accuracies figure
    plt.tight_layout()
    plt.savefig(filename + "_accs.png")
    plt.close()

def plot_embedding(model_file, X=GET_DEFAULTS['X'], y=GET_DEFAULTS['y']):
    '''
    Function to plot umap for model embedding

    :param model_file : model filename
    :param X : X data filepath
    :param y : y data filepath
    '''

    X, y = load_data(X, y)
    base_model = load_model(model_file)
    X = np.reshape(X, [-1, 9, 64, 1])
    model = tf.keras.Sequential(base_model.layers[:-1])
    y_pred = model.predict(X)
    print(y_pred.shape)
    fit = umap.UMAP(n_neighbors=15)
    u = fit.fit_transform(y_pred)
    print(u[:, 0].shape, u[:, 1].shape)
    scatter = plt.scatter(u[:, 0], u[:, 1], c=y, alpha=0.5)
    plt.legend(handles=scatter.legend_elements()[0], labels=TARGET_SECTIONS)
    plt.savefig(model_file.replace(".h5", "_umap.png"))
    plt.close()


if __name__ == "__main__":
    path = '../output/BaseSection_TLGenre_40Epochs\model'
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.h5') and '_TL_G_' in file:
                plot_embedding(os.path.join(path, file))
                print(os.path.join(path, file))

    # plot_embedding('../output/BaseGenre_TLSection/model/Model_G_33_1_22_0.h5')


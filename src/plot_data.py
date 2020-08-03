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


def plot_confusion_matrix(data, label, filename='plot'):
    sb.heatmap(data, annot = True, xticklabels = label, yticklabels = label, cmap="YlGnBu", fmt='g')
    plt.savefig(filename + '_cm.png')
    # plt.show()

def plot_epochs(history, epochs, filename):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
    ax1, ax2 = axes
    ax1.plot(range(epochs), history.history['accuracy'], label = 'Training Accuracy')
    ax1.plot(range(epochs), history.history['val_accuracy'], label = 'Validation Accuracy')
    ax1.set_ylim(0, 1)
    # ax2.set_ylim(0, 1)
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
    # plt.show()

# def umap_plot():
#     X = np.load('../data/Groove_Monkee_Mega_Pack_GM_Full_both_X.npy').astype('float32')
#     X /= np.max(X)
#     X = X.reshape(1834, 576)
#     y1 = np.load('../data/Groove_Monkee_Mega_Pack_GM_Full_both_y1.npy').astype('int')
#     y2 = np.load('../data/Groove_Monkee_Mega_Pack_GM_Full_both_y2.npy').astype('int')
#     fit = umap.UMAP()
#     print(y1)
#     u = fit.fit_transform(X)
#     print(u[:, 0].shape, u[:, 1].shape)
#     plt.scatter(u[:, 0], u[:, 1], c=y2, s =y1)
#     plt.show()



def plot_embedding(model_file, X=GET_DEFAULTS['X'], y=GET_DEFAULTS['y']):
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
    # plt.show()
    # u = fit.fit(y_pred)
    # umap.plot.points(u, labels=y)
    # umap.plot.plt.show()

if __name__ == "__main__":
    path = '../output/BaseSection_TLGenre_40Epochs\model'
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.h5') and '_TL_G_' in file:
                plot_embedding(os.path.join(path, file))
                print(os.path.join(path, file))

    # plot_embedding('../output/BaseGenre_TLSection/model/Model_G_33_1_22_0.h5')


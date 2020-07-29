import numpy as np

import seaborn as sb
import matplotlib.pyplot as plt
import umap
import umap.plot

# # from tensorflow.keras.models import load_model, Model
# # import tensorflow as tf

# from constants import GET_DEFAULTS, TARGET_GENRES
# from utilities import load_data, split_data


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
    ax2.set_ylim(0, 1)
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



# def plot_embedding():
#     X, y = load_data('../data/Groove_Monkee_Mega_Pack_GM_Full_X1.npy', '../data/Groove_Monkee_Mega_Pack_GM_Full_y1.npy')
#     base_model = load_model('../output/Base_SECTION_TL_GENRE/Model_S_94_2_32_0.h5')
#     # y = to_categorical(y)
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     model = tf.keras.Sequential(base_model.layers[:-1])
#     # model.summary()
#     y1 = model.predict(X_train)
#     print(y1.shape)
#     fit = umap.UMAP(n_neighbors=15)
#     # print(y1)
#     u = fit.fit_transform(y1)
#     print(u[:, 0].shape, u[:, 1].shape)
#     plt.scatter(u[:, 0], u[:, 1], c=y_train)
#     plt.show()
#     # umap.plot.points(u, labels=y_train)
#     # umap.plot.plt.show()

if __name__ == "__main__":
    # plot_embedding()

    






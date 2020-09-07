import numpy as np
import os
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import umap
import umap.plot
from sklearn.model_selection  import train_test_split
from sklearn.manifold import TSNE

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
    # plt.figure(figsize=(5,5)) 
    sns.heatmap(data, annot = True, xticklabels = label, yticklabels = label, cmap="YlGnBu", fmt='g')
    plt.xlabel('Predicted Label', labelpad=10)
    plt.ylabel('Actual Label', labelpad=15)
    # plt.axis('square')
    plt.savefig(filename + '_cm.png')
    plt.close()

def plot_curve():
    # loss = [1.3504,1.1596,0.8588,0.6291,0.5003,0.3928,0.3876,0.2765,0.2397,0.1960,0.1723,0.1740,0.1247,0.1044,0.0682,0.0651,0.0802,0.0674,0.0402,0.0747,0.0654,0.0460,0.0593,0.0282,0.0515,0.0507,0.0349,0.0212,0.0185,0.0266,0.0303,0.0819,0.0396,0.0194,0.0194,0.0077,0.0013,0.0055,0.0351,0.0222]
    # accuracy = [0.4054,0.4716,0.6719,0.7526,0.8097,0.8481,0.8546,0.8979,0.9103,0.9305,0.9344,0.9358,0.9555,0.9657,0.9762,0.9784,0.9703,0.9779,0.9857,0.9720,0.9786,0.9844,0.9803,0.9920,0.9815,0.9820,0.9893,0.9944,0.9951,0.9908,0.9876,0.9747,0.9874,0.9937,0.9959,0.9973,0.9998,0.9985,0.9903,0.9922]
    # val_loss = [1.1988,1.0406,0.7768,0.6136,0.4979,0.6532,0.3447,0.2662,0.2408,0.2387,0.2547,0.2114,0.1986,0.1806,0.2157,0.1803,0.2959,0.1790,0.2022,0.2638,0.2614,0.2157,0.1819,0.2254,0.1787,0.1545,0.1680,0.1903,0.2284,0.2547,0.1986,0.1698,0.1591,0.1498,0.1565,0.1829,0.1845,0.1961,0.2282,0.18500]
    # val_accuracy = [0.3926,0.5306,0.6744,0.7619,0.8095,0.7823,0.8698,0.9067,0.9300,0.9096,0.9145,0.9349,0.9329,0.9407,0.9417,0.9456,0.9048,0.9582,0.9475,0.9213,0.9466,0.9388,0.9563,0.9563,0.9524,0.9553,0.9563,0.9621,0.9456,0.9417,0.9485,0.9514,0.9592,0.9621,0.9650,0.9640,0.9670,0.9631,0.9514,0.9543]
    
    
    
    # loss = [9.8433,6.1637,3.4902,2.3516,1.8885,1.4978,1.1478,0.8719,0.7783,0.7545,0.7394,0.7306,0.7270,0.7190,0.7153,0.7113,0.7076,0.7024,0.6989,0.6958,0.6952,0.6933,0.6916,0.6897,0.6840,0.6818,0.6809,0.6791,0.6760,0.6776,0.6780,0.6720,0.6713,0.6713,0.6702,0.6678,0.6676,0.6653,0.6627,0.6615]
    # accuracy = [0.0987,0.1184,0.5078,0.6071,0.7075,0.7075,0.7010,0.7010,0.6794,0.7033,0.7045,0.7087,0.7111,0.6986,0.6776,0.7087,0.6836,0.7045,0.6902,0.6914,0.7045,0.6806,0.7063,0.6836,0.7051,0.6968,0.7111,0.6866,0.7039,0.6878,0.7087,0.6950,0.7123,0.7153,0.7063,0.7063,0.6986,0.7057,0.7075,0.7129]
    # val_loss = [8.4097,4.9176,3.1838,2.3748,1.9693,1.4856,1.1039,0.8853,0.8527,0.8355,0.8222,0.8151,0.8031,0.7962,0.7914,0.7839,0.7769,0.7742,0.7703,0.7679,0.7682,0.7582,0.7551,0.7527,0.7515,0.7487,0.7451,0.7415,0.7396,0.7379,0.7376,0.7343,0.7336,0.7302,0.7280,0.7279,0.7291,0.7260,0.7254,0.7233]
    # val_accuracy = [0.0931,0.1718,0.5346,0.6778,0.6874,0.6683,0.6659,0.6659,0.6683,0.6802,0.6683,0.6850,0.6778,0.6754,0.6635,0.6706,0.6706,0.6850,0.6563,0.6683,0.6730,0.6754,0.6802,0.6778,0.6778,0.6969,0.6969,0.6778,0.6826,0.6874,0.6969,0.6993,0.6969,0.6826,0.7088,0.6897,0.6993,0.6945,0.6993,0.6921]

    # loss = [6.5925, 0.9444, 0.7172, 0.6352, 0.5907, 0.5519, 0.5250, 0.5096, 0.4862, 0.4629, 0.4489, 0.4362, 0.4312, 0.4228, 0.4322, 0.4110, 0.4004, 0.4012, 0.3874, 0.3607, 0.3595, 0.3562, 0.3694, 0.3563, 0.3335, 0.3757, 0.3345, 0.3256, 0.3218, 0.3097, 0.3200, 0.3085, 0.3007, 0.2851, 0.2791, 0.2736, 0.2748, 0.2860, 0.2781, 0.2794]
    # accuracy = [0.2446, 0.6471, 0.7243, 0.7464, 0.7656, 0.7721, 0.7793, 0.7931, 0.7919, 0.8056, 0.8182, 0.8289, 0.8313, 0.8373, 0.8361, 0.8361, 0.8373, 0.8248, 0.8385, 0.8571, 0.8618, 0.8571, 0.8559, 0.8565, 0.8750, 0.8463, 0.8690, 0.8720, 0.8660, 0.8780, 0.8714, 0.8720, 0.8834, 0.8894, 0.8959, 0.8977, 0.8971, 0.8876, 0.8846, 0.8894]
    # val_loss = [1.2361, 0.8138, 0.7020, 0.6553, 0.6362, 0.6100, 0.5909, 0.5787, 0.5587, 0.5567, 0.5461, 0.5444, 0.5395, 0.5514, 0.5489, 0.5237, 0.5073, 0.5134, 0.5161, 0.5098, 0.5086, 0.5080, 0.5274, 0.4912, 0.4911, 0.5385, 0.4912, 0.5099, 0.4877, 0.4791, 0.4847, 0.4877, 0.4915, 0.4763, 0.4943, 0.4775, 0.4953, 0.5506, 0.5496, 0.4643]
    # val_accuracy = [0.6181, 0.6635, 0.7351, 0.7399, 0.7375, 0.7542, 0.7566, 0.7685, 0.7780, 0.7804, 0.7804, 0.7804, 0.7971, 0.7900, 0.7947, 0.7995, 0.8019, 0.8091, 0.8115, 0.8138, 0.8067, 0.7947, 0.8043, 0.8115, 0.8234, 0.8019, 0.8305, 0.8234, 0.8282, 0.8234, 0.8353, 0.8186, 0.8377, 0.8305, 0.8353, 0.8377, 0.8425, 0.8115, 0.8186, 0.8329]

    # loss = [2.0798, 1.6889, 1.1732, 0.8501, 0.6917, 0.5697, 0.4826, 0.4470, 0.4085, 0.4017, 0.3243, 0.3194, 0.3472, 0.2723, 0.2646, 0.2678, 0.2145, 0.1958, 0.1780, 0.2155, 0.2191, 0.1455, 0.1199, 0.1314, 0.0996, 0.1358, 0.1330, 0.1292, 0.0965, 0.0959, 0.0779, 0.0833, 0.0540, 0.0839, 0.0955, 0.1100, 0.0577, 0.0630, 0.0676, 0.0764]
    # genre_loss = [1.0620, 0.7857, 0.4340, 0.2565, 0.2111, 0.1911, 0.1741, 0.1650, 0.1587, 0.1445, 0.1318, 0.1289, 0.1357, 0.1021, 0.0899, 0.1115, 0.0714, 0.0623, 0.0582, 0.0771, 0.0796, 0.0420, 0.0335, 0.0332, 0.0204, 0.0323, 0.0346, 0.0384, 0.0184, 0.0234, 0.0145, 0.0232, 0.0079, 0.0235, 0.0277, 0.0262, 0.0070, 0.0131, 0.0169, 0.0204]
    # section_loss = [1.0178, 0.9032, 0.7393, 0.5936, 0.4806, 0.3786, 0.3084, 0.2820, 0.2498, 0.2571, 0.1925, 0.1905, 0.2115, 0.1702, 0.1748, 0.1563, 0.1432, 0.1335, 0.1198, 0.1384, 0.1394, 0.1035, 0.0863, 0.0983, 0.0792, 0.1035, 0.0984, 0.0908, 0.0781, 0.0725, 0.0634, 0.0601, 0.0460, 0.0604, 0.0678, 0.0838, 0.0507, 0.0498, 0.0506, 0.0560]
    # genre_accuracy = [0.4744, 0.6217, 0.8355, 0.8943, 0.9012, 0.9125, 0.9115, 0.9204, 0.9194, 0.9313, 0.9326, 0.9402, 0.9382, 0.9571, 0.9673, 0.9541, 0.9703, 0.9805, 0.9792, 0.9719, 0.9693, 0.9858, 0.9891, 0.9871, 0.9934, 0.9898, 0.9888, 0.9868, 0.9937, 0.9927, 0.9944, 0.9927, 0.9974, 0.9924, 0.9934, 0.9921, 0.9977, 0.9947, 0.9941, 0.9944]
    # section_accuracy = [0.4592, 0.5362, 0.6204, 0.7076, 0.7767, 0.8553, 0.8890, 0.8933, 0.9082, 0.9042, 0.9343, 0.9323, 0.9224, 0.9415, 0.9339, 0.9442, 0.9485, 0.9508, 0.9551, 0.9498, 0.9478, 0.9623, 0.9666, 0.9653, 0.9686, 0.9600, 0.9623, 0.9660, 0.9683, 0.9732, 0.9762, 0.9775, 0.9822, 0.9756, 0.9739, 0.9722, 0.9782, 0.9822, 0.9815, 0.9785]
    # val_loss = [2.2315, 1.5861, 1.0109, 0.8050, 0.7486, 0.5767, 0.6514, 0.5473, 0.5335, 0.5236, 0.4611, 0.6589, 0.5098, 0.4352, 0.5299, 0.4330, 0.5015, 0.4757, 0.5190, 0.5465, 0.3762, 0.4951, 0.4284, 0.3896, 0.3998, 0.4803, 0.5687, 0.4408, 0.4849, 0.4799, 0.3768, 0.4679, 0.4727, 0.4812, 0.4584, 0.4285, 0.3711, 0.4148, 0.3133, 0.4060]
    # val_genre_loss = [1.2232, 0.6738, 0.3579, 0.2450, 0.2402, 0.2178, 0.2043, 0.2098, 0.1929, 0.1917, 0.1795, 0.2288, 0.1713, 0.1506, 0.1484, 0.1510, 0.1109, 0.1059, 0.1459, 0.1451, 0.0821, 0.1091, 0.0715, 0.0787, 0.1109, 0.0947, 0.1305, 0.1338, 0.0722, 0.0816, 0.0730, 0.0804, 0.0728, 0.0655, 0.0904, 0.0449, 0.0452, 0.0765, 0.0391, 0.0479]
    # val_section_loss = [1.0083, 0.9123, 0.6530, 0.5600, 0.5085, 0.3589, 0.4470, 0.3375, 0.3406, 0.3319, 0.2816, 0.4301, 0.3385, 0.2846, 0.3815, 0.2820, 0.3906, 0.3698, 0.3731, 0.4014, 0.2941, 0.3860, 0.3569, 0.3109, 0.2889, 0.3855, 0.4382, 0.3069, 0.4128, 0.3982, 0.3037, 0.3875, 0.3999, 0.4157, 0.3680, 0.3836, 0.3259, 0.3383, 0.2741, 0.3581]
    # val_genre_accuracy = [0.5015, 0.7923, 0.8546, 0.8813, 0.8872, 0.8635, 0.9080, 0.8872, 0.8991, 0.9228, 0.9080, 0.9347, 0.9139, 0.9555, 0.9496, 0.9407, 0.9614, 0.9674, 0.9407, 0.9525, 0.9703, 0.9585, 0.9674, 0.9822, 0.9733, 0.9763, 0.9674, 0.9496, 0.9763, 0.9792, 0.9763, 0.9852, 0.9881, 0.9792, 0.9763, 0.9911, 0.9881, 0.9852, 0.9881, 0.9881]
    # val_section_accuracy = [0.4985, 0.5846, 0.6914, 0.7240, 0.8131, 0.8843, 0.8279, 0.8576, 0.8783, 0.8694, 0.8872, 0.8546, 0.8813, 0.8991, 0.8932, 0.9021, 0.8902, 0.8843, 0.8902, 0.8576, 0.9110, 0.9050, 0.9199, 0.9318, 0.9050, 0.9080, 0.9080, 0.9080, 0.8991, 0.9228, 0.9169, 0.9318, 0.8961, 0.8932, 0.9169, 0.9169, 0.9169, 0.9110, 0.9288, 0.9110] 

    # loss = [4.3257, 2.3925, 1.8887, 1.6434, 1.5120, 1.3977, 1.3087, 1.2450, 1.1917, 1.1749, 1.1696, 1.1648, 1.1609, 1.1588, 1.1548, 1.1544, 1.1552, 1.1531, 1.1501, 1.1454, 1.1443, 1.1484, 1.1412, 1.1399, 1.1382, 1.1354, 1.1344, 1.1349, 1.1356, 1.1319, 1.1327, 1.1289, 1.1291, 1.1296, 1.1263, 1.1240, 1.1236, 1.1296, 1.1222, 1.1237]
    # val_loss = [3.8167, 2.8062, 2.3229, 2.0141, 1.7999, 1.5849, 1.4351, 1.2838, 1.2088, 1.1988, 1.1921, 1.1860, 1.1821, 1.1827, 1.1755, 1.1765, 1.1682, 1.1752, 1.1710, 1.1674, 1.1708, 1.1644, 1.1640, 1.1642, 1.1610, 1.1566, 1.1574, 1.1563, 1.1515, 1.1552, 1.1547, 1.1562, 1.1512, 1.1490, 1.1502, 1.1525, 1.1647, 1.1444, 1.1494, 1.1559]
    # accuracy = [0.0938, 0.4071, 0.4118, 0.4071, 0.4098, 0.4288, 0.4295, 0.4237, 0.4363, 0.4290, 0.4417, 0.4443, 0.4441, 0.4504, 0.4548, 0.4504, 0.4575, 0.4562, 0.4570, 0.4572, 0.4584, 0.4584, 0.4623, 0.4609, 0.4628, 0.4638, 0.4638, 0.4679, 0.4628, 0.4701, 0.4640, 0.4660, 0.4720, 0.4674, 0.4699, 0.4718, 0.4689, 0.4691, 0.4689, 0.4682]
    # val_accuracy = [0.2255, 0.4043, 0.3994, 0.4004, 0.4121, 0.4247, 0.4198, 0.4169, 0.4189, 0.4295, 0.4422, 0.4470, 0.4519, 0.4519, 0.4606, 0.4655, 0.4645, 0.4723, 0.4606, 0.4704, 0.4655, 0.4606, 0.4548, 0.4665, 0.4655, 0.4772, 0.4752, 0.4810, 0.4869, 0.4636, 0.4713, 0.4674, 0.4772, 0.4733, 0.4742, 0.4655, 0.4665, 0.4849, 0.4772, 0.4752]

    loss = [2.0874, 1.1082, 1.0481, 1.0099, 0.9827, 0.9419, 0.9285, 0.8988, 0.8606, 0.8364, 0.8224, 0.7988, 0.7666, 0.7532, 0.7344, 0.7309, 0.7528, 0.7181, 0.7148, 0.6835, 0.6631, 0.6407, 0.6581, 0.6401, 0.6300, 0.6305, 0.6428, 0.6190, 0.6170, 0.6195, 0.6001, 0.6116, 0.5862, 0.5691, 0.5770, 0.5926, 0.5740, 0.5924, 0.5569, 0.5771]
    val_loss = [1.2293, 1.0876, 1.0358, 1.0058, 0.9530, 0.9761, 0.9000, 0.9105, 0.8819, 0.8677, 0.8062, 0.7861, 0.7811, 0.7578, 0.7669, 0.7263, 0.7607, 0.7342, 0.7229, 0.6871, 0.6801, 0.7489, 0.6739, 0.7018, 0.6591, 0.7047, 0.6548, 0.6636, 0.6642, 0.6215, 0.6640, 0.6852, 0.6228, 0.6368, 0.6474, 0.7294, 0.7246, 0.6151, 0.6684, 0.6187]
    accuracy = [0.3787, 0.4971, 0.5287, 0.5438, 0.5656, 0.5890, 0.5926, 0.6060, 0.6330, 0.6291, 0.6517, 0.6578, 0.6743, 0.6852, 0.6920, 0.6920, 0.6818, 0.6957, 0.7005, 0.7103, 0.7175, 0.7331, 0.7263, 0.7224, 0.7360, 0.7385, 0.7304, 0.7421, 0.7380, 0.7365, 0.7496, 0.7467, 0.7589, 0.7632, 0.7611, 0.7543, 0.7635, 0.7533, 0.7713, 0.7664]
    val_accuracy = [0.4937, 0.5199, 0.5481, 0.5423, 0.5977, 0.5821, 0.6443, 0.6229, 0.6365, 0.6511, 0.6696, 0.6851, 0.6842, 0.6832, 0.6395, 0.6929, 0.6842, 0.6754, 0.6880, 0.7094, 0.7065, 0.6725, 0.7143, 0.7075, 0.7230, 0.6919, 0.7085, 0.7182, 0.7191, 0.7473, 0.7405, 0.7182, 0.7629, 0.7366, 0.7512, 0.6968, 0.6929, 0.7658, 0.7716, 0.7551]

    plt.figure(figsize=(5,5)) 
    sns.set_style("ticks")
    # plt.axis('square')
    sns.lineplot(range(40), loss, label = 'Training Loss')
    sns.lineplot(range(40), val_loss, label = 'Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    
    plt.savefig('genre_tl_33_0_22_5_loss.png')
    plt.close()
    plt.figure(figsize=(5,5)) 
    sns.set_style("ticks")
    sns.lineplot(range(40), accuracy, label = 'Training Accuracy')
    sns.lineplot(range(40), val_accuracy, label = 'Validation Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.xlabel('Number of Epochs')
    # plt.axis('square')
    plt.ylabel('Accuracy')
    plt.savefig('genre_tl_33_0_22_5_accuracy.png')
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

def plot_scatter(X, y , model_path, target_list, filename):

    X, y = load_data(X, y) 
    base_model = load_model(model_path)
    X = np.reshape(X, [-1, 9, 64, 1])
    model = tf.keras.Sequential(base_model.layers[:-1])
    y_pred = model.predict(X)
    # print(y_pred.shape)

    sns.set_style("ticks")

    u = umap.UMAP(min_dist=0.5, n_neighbors=15)
    umap_obj = u.fit_transform(y_pred)
    umap_df = pd.DataFrame({'X1':umap_obj[:,0],
                        'X2':umap_obj[:,1],
                        'y':y})
    

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(y_pred)
    tsne_df = pd.DataFrame({'X1':tsne_obj[:,0],
                        'X2':tsne_obj[:,1],
                        'y':y})    
    
    for i in range(10):
        tsne_df_filter = pd.concat([
            tsne_df[tsne_df['y'] == 0].sample(n=100, random_state=i),
            tsne_df[tsne_df['y'] == 1].sample(n=100, random_state=i),
            tsne_df[tsne_df['y'] == 2].sample(n=100, random_state=i),
            tsne_df[tsne_df['y'] == 3].sample(n=100, random_state=i),
            tsne_df[tsne_df['y'] == 4].sample(n=100, random_state=i)])
        tsne_df_filter['y_label'] = [target_list[label] for label in tsne_df_filter['y']]
    
        umap_df_filter = pd.concat([
            umap_df[umap_df['y'] == 0].sample(n=100, random_state=i),
            umap_df[umap_df['y'] == 1].sample(n=100, random_state=i),
            umap_df[umap_df['y'] == 2].sample(n=100, random_state=i),
            umap_df[umap_df['y'] == 3].sample(n=100, random_state=i),
            umap_df[umap_df['y'] == 4].sample(n=100, random_state=i)])
        umap_df_filter['y_label'] = [target_list[label] for label in umap_df_filter['y']]

        fig, ax = plt.subplots()
        g = sns.scatterplot(x="X1", y="X2", hue="y_label", palette="Set2", size =1,
                            data=tsne_df_filter, ax=ax
                    # style="Section", 
                    # markers=markers,
                    # edgecolor="black",
                    # edgewidth=0.5,
                    )
    
        handles, labels = ax.get_legend_handles_labels()
        plt.axis('square')
        print(handles, labels)  
        plt.legend(handles=handles[1:6], labels=labels[1:6], bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../output/plots/' + filename + '_tsne_' + str(i) + '.png')
        plt.close()
        fig, ax = plt.subplots()
        g = sns.scatterplot(x="X1", y="X2", hue="y_label", palette="Set2", size =1,
                            data=umap_df_filter, ax=ax
                    # style="Section", 
                    # markers=markers,
                    # edgecolor="black",
                    # edgewidth=0.5,
                    )
    
        handles, labels = ax.get_legend_handles_labels()
        plt.axis('square')
        print(handles, labels)  
        plt.legend(handles=handles[1:6], labels=labels[1:6], bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../output/plots/' + filename + '_umap_' + str(i) + '.png')
        plt.close()

def plot_scatter_mo(X, y1 ,y2, model_path, target_list1, target_list2, filename):

    X, y1, y2 = load_data(X, y1, y2) 
    base_model = load_model(model_path)
    X = np.reshape(X, [-1, 9, 64, 1])
    model = tf.keras.Sequential(base_model.layers[:-1])
    y_pred = model.predict(X)
    # print(y_pred.shape)

    sns.set_style("ticks")

    u = umap.UMAP(min_dist=0.5, n_neighbors=15)
    umap_obj = u.fit_transform(y_pred)
    umap_df = pd.DataFrame({'X1':umap_obj[:,0],
                        'X2':umap_obj[:,1],
                        'y1':y1,
                        'y2':y2})
    

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj= tsne.fit_transform(y_pred)
    tsne_df = pd.DataFrame({'X1':tsne_obj[:,0],
                        'X2':tsne_obj[:,1],
                        'y1':y1,
                        'y2':y2})    
    
    for i in range(10):
        tsne_df_filter = pd.concat([
            tsne_df[tsne_df['y1'] == 0].sample(n=20, random_state=i),
            tsne_df[tsne_df['y1'] == 1].sample(n=20, random_state=i),
            tsne_df[tsne_df['y1'] == 2].sample(n=20, random_state=i),
            tsne_df[tsne_df['y1'] == 3].sample(n=20, random_state=i),
            tsne_df[tsne_df['y1'] == 4].sample(n=20, random_state=i),
            tsne_df[tsne_df['y2'] == 0].sample(n=50, random_state=i),
            tsne_df[tsne_df['y2'] == 1].sample(n=50, random_state=i),
            tsne_df[tsne_df['y2'] == 2].sample(n=50, random_state=i),
            tsne_df[tsne_df['y2'] == 3].sample(n=50, random_state=i),
            tsne_df[tsne_df['y2'] == 4].sample(n=50, random_state=i)])
        tsne_df_filter['Genre'] = [target_list1[label] for label in tsne_df_filter['y1']]
        tsne_df_filter['Section'] = [target_list2[label] for label in tsne_df_filter['y2']]
    
        umap_df_filter = pd.concat([
            umap_df[umap_df['y1'] == 0].sample(n=20, random_state=i),
            umap_df[umap_df['y1'] == 1].sample(n=20, random_state=i),
            umap_df[umap_df['y1'] == 2].sample(n=20, random_state=i),
            umap_df[umap_df['y1'] == 3].sample(n=20, random_state=i),
            umap_df[umap_df['y1'] == 4].sample(n=20, random_state=i),
            umap_df[umap_df['y2'] == 0].sample(n=50, random_state=i),
            umap_df[umap_df['y2'] == 1].sample(n=50, random_state=i),
            umap_df[umap_df['y2'] == 2].sample(n=50, random_state=i),
            umap_df[umap_df['y2'] == 3].sample(n=50, random_state=i),
            umap_df[umap_df['y2'] == 4].sample(n=50, random_state=i)])
        umap_df_filter['Genre'] = [target_list1[label] for label in umap_df_filter['y1']]
        umap_df_filter['Section'] = [target_list2[label] for label in umap_df_filter['y2']]
        markers = ['o', 'v', 's', 'P', 'D']

        # fig, ax = plt.subplots()
        # g = sns.scatterplot(x="X1", y="X2", hue="Genre", palette="Set2", #size =1,
        #                     data=tsne_df_filter, ax=ax,
        #                     style="Section", 
        #                     # edgecolor="black",
        #                     markers=markers
        #             )
        # handles, labels = ax.get_legend_handles_labels()
        # # labels[6] = '----'
        # plt.axis('square')
        # print(handles, labels)  
        # plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc='upper left')
        # plt.xlabel('')
        # plt.ylabel('')
        # plt.savefig('../output/plots/' + filename + '_tsne_' + str(i) + '.png')
        # plt.close()
        # fig, ax = plt.subplots()
        # g = sns.scatterplot(x="X1", y="X2", hue="Genre", palette="Set2", #size =1,
        #                     data=umap_df_filter, ax=ax,
        #                     style="Section", 
        #                     # edgecolor="black",
        #                     markers=markers
        #             )
    
        # handles, labels = ax.get_legend_handles_labels()
        # # labels[6] = '----'
        # plt.axis('square')
        # print(handles, labels)  
        # plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc='upper left')
        # plt.xlabel('')
        # plt.ylabel('')
        # plt.savefig('../output/plots/' + filename + '_umap_' + str(i) + '.png')
        # plt.close()

        fig, ax = plt.subplots()
        g = sns.scatterplot(x="X1", y="X2", hue="Genre", palette="Set2", size =1,
                            data=tsne_df_filter, ax=ax
                            # style="Section", 
                            # edgecolor="black",
                            # markers=markers
                    )
        handles, labels = ax.get_legend_handles_labels()
        # labels[6] = '----'
        plt.axis('square')
        print(handles, labels)  
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../output/plots/' + filename + '_tsne_1_' + str(i) + '.png')
        plt.close()
        fig, ax = plt.subplots()
        g = sns.scatterplot(x="X1", y="X2", hue="Genre", palette="Set2", size =1,
                            data=umap_df_filter, ax=ax
                            # style="Section", 
                            # edgecolor="black",
                            # markers=markers
                    )
    
        handles, labels = ax.get_legend_handles_labels()
        plt.axis('square')
        print(handles, labels)  
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../output/plots/' + filename + '_umap_1_' + str(i) + '.png')
        plt.close()

        fig, ax = plt.subplots()
        g = sns.scatterplot(x="X1", y="X2", hue="Section", palette="Set2", size =1,
                            data=tsne_df_filter, ax=ax
                            # style="Section", 
                            # edgecolor="grey",
                            # markers=markers
                    )
        handles, labels = ax.get_legend_handles_labels()
        # labels[6] = '----'
        plt.axis('square')
        print(handles, labels)  
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../output/plots/' + filename + '_tsne_2_' + str(i) + '.png')
        plt.close()
        fig, ax = plt.subplots()
        g = sns.scatterplot(x="X1", y="X2", hue="Section", palette="Set2", size =1,
                            data=umap_df_filter, ax=ax,
                            # style="Section", 
                            # edgecolor="grey",
                            # markers=markers
                    )
    
        handles, labels = ax.get_legend_handles_labels()
        plt.axis('square')
        print(handles, labels)  
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1, 1), loc='upper left')
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig('../output/plots/' + filename + '_umap_2_' + str(i) + '.png')
        plt.close()

        

if __name__ == "__main__":
    # path = '../output/BaseSection_TLGenre_40Epochs\model'
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         if file.endswith('.h5') and '_TL_G_' in file:
    #             plot_embedding(os.path.join(path, file))
    #             print(os.path.join(path, file))

    # plot_embedding('../output/BaseGenre_TLSection/model/Model_G_33_1_22_0.h5')
    # plot_curve()
    # data_genre =   [[0.94, 0.06, 0.  , 0.  , 0.  ],
    #                 [0.  , 1.  , 0.  , 0.  , 0.  ],
    #                 [0.01, 0.  , 0.94, 0.01, 0.04],
    #                 [0.  , 0.01, 0.01, 0.94, 0.03],
    #                 [0.01, 0.01, 0.  , 0.  , 0.98]]
    
    # data_section = [[0.93, 0.  , 0.05, 0.  , 0.02],
    #                 [0.  , 0.98, 0.02, 0.  , 0.  ],
    #                 [0.12, 0.04, 0.77, 0.08, 0.  ],
    #                 [0.  , 0.18, 0.  , 0.82, 0.  ],
    #                 [0.06, 0.  , 0.08, 0.  , 0.86]]
    # data_section_tl_0 = [[0.77, 0.02, 0.05, 0.  , 0.16],
    #                     [0.03, 0.91, 0.02, 0.  , 0.04],
    #                     [0.19, 0.27, 0.46, 0.  , 0.08],
    #                     [0.64, 0.18, 0.09, 0.  , 0.09],
    #                     [0.75, 0.03, 0.03, 0.  , 0.19]]
    # data_section_tl_10 = [[0.74, 0.  , 0.02, 0.  , 0.23],
    #                         [0.01, 0.97, 0.02, 0.01, 0.  ],
    #                         [0.12, 0.04, 0.73, 0.04, 0.08],
    #                         [0.09, 0.27, 0.09, 0.55, 0.  ],
    #                         [0.28, 0.  , 0.06, 0.  , 0.67]]
    # data_genre_tl_0 = [[0.48 ,0.39 ,0.02 ,0.02 ,0.09],
    #                     [0.04 ,0.82 ,0.02 ,0.02 ,0.09],
    #                     [0.   ,0.01 ,0.75 ,0.05 ,0.19],
    #                     [0.   ,0.   ,0.4  ,0.54 ,0.06],
    #                     [0.02 ,0.02 ,0.03 ,0.01 ,0.92]]
    
    # data_genre_tl_10 = [[0.53, 0.22, 0.25, 0.  , 0.  ],
    #                     [0.36, 0.47, 0.13, 0.02, 0.02],
    #                     [0.06, 0.02, 0.18, 0.17, 0.57],
    #                     [0.13, 0.03, 0.46, 0.29, 0.1 ],
    #                     [0.1 , 0.01, 0.12, 0.03, 0.74]]
    # data_mo_genre =[[0.94, 0.  , 0.03, 0.  , 0.03],
    #                 [0.04, 0.96, 0.  , 0.  , 0.  ],
    #                 [0.  , 0.  , 1.  , 0.  , 0.  ],
    #                 [0.  , 0.  , 0.  , 1.  , 0.  ],
    #                 [0.  , 0.  , 0.  , 0.  , 1.  ]]
    # data_mo_section =[[0.82, 0.  , 0.03, 0.03, 0.12],
    #                     [0.  , 0.92, 0.01, 0.  , 0.07],
    #                     [0.17, 0.17, 0.42, 0.08, 0.17],
    #                     [0.  , 0.  , 0.  , 1.  , 0.  ],
    #                     [0.01, 0.02, 0.  , 0.01, 0.96]]
    data =  [[0.83, 0.03, 0.  , 0.  , 0.14],
 [0.58, 0.2 , 0.04, 0.  , 0.18],
 [0.  , 0.  , 0.54, 0.01, 0.45],
 [0.01, 0.  , 0.64, 0.06, 0.29],
 [0.11, 0.04, 0.17, 0.  , 0.67]]
    
    plot_confusion_matrix(data, TARGET_GENRES, 'genre_tl0')
    # plot_confusion_matrix(data_section, TARGET_SECTIONS, 'section')
    # plot_confusion_matrix(data_section_tl_0, TARGET_SECTIONS, 'section_tl0')
    # plot_confusion_matrix(data_section_tl_10, TARGET_SECTIONS, 'section_tl10')
    # plot_confusion_matrix(data_genre_tl_0, TARGET_GENRES, 'genre_tl0')
    # plot_confusion_matrix(data_genre_tl_10, TARGET_GENRES, 'genre_tl10')
    # plot_confusion_matrix(data_mo_genre, TARGET_GENRES, 'mo_genre')
    # plot_confusion_matrix(data_mo_section, TARGET_SECTIONS, 'mo_section')
    # plot_scatter(GET_DEFAULTS["X"], GET_DEFAULTS["y"], '../output/BaseGenre_TLSection/model/Model_G_33_1_22_0.h5', TARGET_GENRES, 'genre')
    # plot_scatter(GET_DEFAULTS["X1"], GET_DEFAULTS["y1"], '../output/BaseSection_TLGenre_40Epochs/model/Model_S_44_2_22_0.h5', TARGET_SECTIONS, 'section')
    # plot_scatter(GET_DEFAULTS["X1"], GET_DEFAULTS["y1"], '../output/BaseGenre_TLSection/model/Model_TL_S_33_1_22_5_0.h5', TARGET_SECTIONS, 'section_tl_0')
    # plot_scatter(GET_DEFAULTS["X1"], GET_DEFAULTS["y1"], '../output/BaseGenre_TLSection/model/Model_TL_S_33_1_22_5_10.h5', TARGET_SECTIONS, 'section_tl_10')
    # plot_scatter(GET_DEFAULTS["X"], GET_DEFAULTS["y"], '../output/BaseSection_TLGenre_40Epochs/model/Model_TL_G_33_1_22_5.h5', TARGET_GENRES, 'genre_tl_0')
    # plot_scatter(GET_DEFAULTS["X"], GET_DEFAULTS["y"], '../output/BaseSection_TLGenre_40Epochs/model/Model_TL_G_33_1_22_5_10.h5', TARGET_GENRES, 'genre_tl_10')
    # plot_scatter_mo(GET_DEFAULTS["X_both"], GET_DEFAULTS["y1_both"], GET_DEFAULTS["y2_both"], '../output/MultiOutput_3/Model_MO_33_1_22_0.h5', TARGET_GENRES, TARGET_SECTIONS, 'mo')
    # plot_curve()
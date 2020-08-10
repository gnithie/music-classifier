import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

"""
A file for evaluation functions. 
"""

def confusion_mtx(actual, prediction, categorical = True):
    '''
    function to rtreive confusion matrix

    :param actual : actual target values
    :param prediction : predicted target values from model
    :param categorical : If its true, argmax will be retrived to perform confusion matrix. By default, this value is set to true.

    :returns confusion matrix 
    '''

    if categorical:
        actual = np.argmax(actual, axis =1)
        prediction = np.argmax(prediction, axis =1)
    conf_mat = confusion_matrix(actual, prediction)
    conf_mat = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals = 2)
    print(conf_mat)
    # plot_confusion_matrix(cmn, TARGET_GENRES)
    return conf_mat

def f_score(actual, prediction, categorical = True):
    '''
    function to retreive f score

    :param actual : actual target values
    :param prediction : predicted target values from model
    :param categorical : If its true, argmax will be retrived to perform confusion matrix. By default, this value is set to true.

    :returns precision, recall, f score and classification report which is class wise precision, recall and f score values
    '''
    if categorical:
        actual = np.argmax(actual, axis =1)
        prediction = np.argmax(prediction, axis =1)
    f1_score = precision_recall_fscore_support(actual, prediction, average='macro')
    print(f1_score)
    cls_report = classification_report(actual, prediction, digits=4)
    cls_report
    print(cls_report)
    return f1_score, cls_report


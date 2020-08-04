import numpy as np

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

def confusion_mtx(actual, prediction, categorical = True):
    if categorical:
        actual = np.argmax(actual, axis =1)
        prediction = np.argmax(prediction, axis =1)
    conf_mat = confusion_matrix(actual, prediction)
    conf_mat = np.around(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis], decimals = 2)
    print(conf_mat)
    # plot_confusion_matrix(cmn, TARGET_GENRES)
    return conf_mat

def f_score(actual, prediction, categorical = True):
    if categorical:
        actual = np.argmax(actual, axis =1)
        prediction = np.argmax(prediction, axis =1)
    f1_score = precision_recall_fscore_support(actual, prediction, average='macro')
    print(f1_score)
    cls_report = classification_report(actual, prediction, digits=4)
    cls_report
    print(cls_report)
    return f1_score, cls_report


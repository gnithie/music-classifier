import numpy as np
from collections import Counter

from sklearn.model_selection  import train_test_split


"""
A utility file to perform general operations
"""

def load_data(filename_X, filename_y, filename_y2 = None):
    '''
    Funtion to load numpy data

    :param filename_X : filepath of the X data
    :param filename_y : filepath of the y data
    :param filename_y2 : filepath of the secondary y data. By default, None.

    :returns X,y data
    '''

    X = np.load(filename_X).astype('float32')
    X /= np.max(X)

    y = np.load(filename_y).astype('int')
    
    if filename_y2 != None:
        y2 = np.load(filename_y2).astype('int')
        return X, y, y2
    return X, y

def split_data(X, y):
    '''
    Function to split X,y data into Train and Test data

    :param X : X data to split
    :param y : y data to split

    :returns X, y data of Train and Test
    '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state = 42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train = np.reshape(X_train, [-1, 9, 64, 1])
    X_test = np.reshape(X_test, [-1, 9, 64, 1])
    return X_train, X_test, y_train, y_test
    
def write_csv(result, filename='result'):
    '''
    Function to write log data into csv file

    :param result : log data
    :filename : csv filename
    '''
    
    res = ','.join(str(v) for v in result)
    f= open('../output/' + filename + '.csv',"a+")
    f.write(res + '\n')
    f.close()
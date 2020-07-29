import numpy as np
from collections import Counter

from sklearn.model_selection  import train_test_split

def load_data(filename_X, filename_y):
    X = np.load(filename_X).astype('float32')
    X /= np.max(X)

    y = np.load(filename_y).astype('int')
    print(Counter(y))
    
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state = 42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train = np.reshape(X_train, [-1, 9, 64, 1])
    X_test = np.reshape(X_test, [-1, 9, 64, 1])
    return X_train, X_test, y_train, y_test
    
def write_csv(result, filename='result'):
    res = ','.join(str(v) for v in result)
    f= open('../output/' + filename + '.csv',"a+")
    f.write(res + '\n')
    f.close()
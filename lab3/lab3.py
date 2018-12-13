import pandas as pd
import numpy as np
from minimization import *

# link = https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease
if __name__ == "__main__":
    column = np.arange(25)
    feature = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod',
               'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
    # for ckd.arff skiprows = 29
    x = pd.read_csv("ckd/chronic_kidney_disease_full.arff",
                    sep=',', na_values=['?', '\t?'], skiprows=145, header=None, usecols=column, names=feature)
    data_polished = x.replace({'normal': 1, ' normal': 1, '\tnormal': 1, 'normal\t': 1,
                               'abnormal': 0, ' abnormal': 0, '\tabnormal': 0, 'abnormal\t': 0,
                               'present': 1, ' present': 1, '\tpresent': 1, 'present\t': 1,
                               'notpresent': 0, ' notpresent': 0, '\tnotpresent': 0, 'notpresent\t': 0,
                               'yes': 1, ' yes': 1, '\tyes': 1, 'yes\t': 1,
                               'no': 0, ' no': 0, '\tno': 0, 'no\t': 0,
                               'good': 1, ' good': 1, '\tgood': 1, 'good\t': 1,
                               'poor': 0, ' poor': 0, '\tpoor': 0, 'poor\t': 0,
                               'ckd': 1, ' ckd': 1, '\tckd': 1, 'ckd\t': 1,
                               'notckd': 0, ' notckd': 0, '\tnotckd': 0, 'notckd\t': 0})

    X_0 = data_polished.dropna(thresh=25).values.astype(float)
    X_5 = data_polished.dropna(thresh=20).values.astype(float)
    Np, Nf = np.shape(X_5)

    mean = np.mean(X_0, axis=0)
    std = np.std(X_0, axis=0)
    mean = mean.reshape(1, Nf)
    std = std.reshape(1, Nf)
    X_norm = (X_0 - mean)/std

    j = 0
    for array in X_5:
        i = 0
        j += 1
        F0 = []
        # print("array: ", array.shape[0])
        # print("X_1 shape: ", X_1.shape[1])
        for i in range(X_5.shape[1]):
            # print(array[i])
            if np.isnan(array[i]):
                F0.append(i)
                # F0 are index of the y columns
        if len(F0) != 0:
            y_train = X_norm[:, F0]
            X_train = np.delete(X_norm, F0, 1)
            ridge = SolveRidge(y_train, X_train)
            w = ridge.run(lamb=10)
            tmp_mean = np.copy(np.delete(mean, F0))
            tmp_std = np.copy(np.delete(std, F0))
            # tmp_mean = tmp_mean.reshape(tmp_mean.shape[0], 1)
            # tmp_std = tmp_std.reshape(tmp_std.shape[0], 1)
            array_train = (array[~np.isnan(array)] - tmp_mean) / tmp_std
            y_hat = np.dot(array_train, w) * std[0, F0] + mean[0, F0]
            array_to_copy = np.copy(array)
            for index in range(len(F0)):
                array_to_copy[F0[index]] = y_hat[index]
            X_5[j-1] = array_to_copy
    # rounding the values
    X = pd.DataFrame(X_5,columns=feature)
    # extend our method
    print("----END----")


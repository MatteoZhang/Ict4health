import pandas as pd
import numpy as np

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
    X_1 = data_polished.dropna(thresh=24).values.astype(float)
    X_2 = data_polished.dropna(thresh=23).values.astype(float)
    X_3 = data_polished.dropna(thresh=22).values.astype(float)
    X_4 = data_polished.dropna(thresh=21).values.astype(float)
    X_5 = data_polished.dropna(thresh=20).values.astype(float)
    Np, Nf = np.shape(X_5)
    y_train = np.zeros(25, dtype=float)

    mean = np.mean(X_0, axis=0)
    std = np.std(X_0, axis=0)
    mean = mean.reshape(1, Nf)
    std = std.reshape(1, Nf)
    X_norm = (X_0 - mean)/std

    data_train = np.zeros((X_1.shape[0], Nf), dtype=float)
    data_train = X_1.copy()

    # find FO the index we want to regress
    # data_train then normalize and for cicle for regression




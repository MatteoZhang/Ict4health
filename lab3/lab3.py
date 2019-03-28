import pandas as pd
import numpy as np
import sklearn.tree as tree
from minimization import *
import graphviz


# link = https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease
def final_round(df):
    df[df < 0] = 0
    i = 0
    for row in df['sg']:
        if row not in [1.005, 1.010, 1.015, 1.020, 1.025]:
            if row < (1.005 + 1.010)/2:
                df.loc[i, 'sg'] = 1.005
            elif row < (1.010 + 1.015)/2:
                df.loc[i, 'sg'] = 1.010
            elif row < (1.015 + 1.020)/2:
                df.loc[i, 'sg'] = 1.015
            elif row < (1.020 + 1.025)/2:
                df.loc[i, 'sg'] = 1.020
            else:
                df.loc[i, 'sg'] = 1.025
        # print(df.loc[i, 'sg']) check
        i += 1
    return df


if __name__ == "__main__":
    column = np.arange(25)
    feature = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod',
               'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
    rounding = [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
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

    df_X = pd.DataFrame(X_5, columns=feature).astype(float)
    decimal = pd.Series(rounding, index=feature)
    df_X_rounded = df_X.round(decimal)
    df_X_final_round = final_round(df_X_rounded)

    data = df_X_final_round.iloc[:, 0:24]
    target = df_X_final_round['class']
    clf = tree.DecisionTreeClassifier("entropy")
    clf = clf.fit(data, target)

    dot_data = tree.export_graphviz(clf, out_file="Tree.dot",
                                    feature_names=feature[0:-1],
                                    class_names=['not ckd', 'ckd'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    # download it https://graphviz.gitlab.io/_pages/Download/Download_windows.html
    # C:\Program Files (x86)\Graphviz2.38\bin\dot.exe
    # C:\Program Files (x86)\Graphviz2.38\bin>dot.exe -Tpng Tree.dot > Tree.png
    # Tree is in the same folder as dot.exe also Tree.dot should be the same
    array = clf.feature_importances_
    print("feature importance:  ", array, "\n")
    print("----END----")


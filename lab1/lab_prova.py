import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sub.min import *

if __name__ == "__main__":
    x = pd.read_csv("parkinsons_updrs.data")
    x.info()
    x.describe()
    x.plot()
    realdata = x.values
    np.random.shuffle(realdata)

    data = realdata[:, 4:22]
    Np, Nf = np.shape(data)

    data_train = data[0:int(Np/2), :]
    data_val = data[int(Np/2):int(Np*0.75), :]
    data_test = data[int(Np*0.75):Np, :]

    mean = np.mean(data_train, 0)
    std = np.std(data_train, 0)

    data_train_norm = (data_train - mean)/std
    data_val_norm = (data_val - mean)/std
    data_test_norm = (data_test - mean)/std

    mean_check = np.mean(data_train_norm, 0)
    std_check = np.std(data_train_norm, 0)

    F0 = 1
    y_train = data_train_norm[:, F0]
    X_train = np.delete(data_train_norm, F0, 1)
    y_val = data_val_norm[:, F0]
    X_val = np.delete(data_val_norm, F0, 1)
    y_test = data_test_norm[:, F0]
    X_test = np.delete(data_test_norm, F0, 1)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    np.random.seed(1)
    logx = 0
    logy = 0
    Nit = 100
    gamma = 1e-5
    lamb = 0.3

    m = SolveLLS(y_train, X_train, y_val, X_val)
    m.run()
    m.print_result('LLS')
    m.plot_w('LLS')

    g = SolveGrad(y_train, X_train, y_val, X_val)
    g.run(gamma, Nit)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)
    g.print_hat('yhat_train vs y_train for Gradient', 'y_train', 'yhat_train', y_train, X_train)
    g.print_hat('yhat_val vs y_val for Gradient', 'y_val', 'yhat_val', y_val, X_val)

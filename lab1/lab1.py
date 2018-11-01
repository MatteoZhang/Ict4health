import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sub.min import *

if __name__ == "__main__":
    x = pd.read_csv("parkinsons_updrs.cvs")
    x.info()
    x.describe()
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
    logy = 1
    Nit = 100
    gamma = 1e-5
    lamb = 0

    m = SolveLLS(y_train, X_train, y_val, X_val)
    m.run()
    m.print_result('LLS')

    g = SolveGrad(y_train, X_train, y_val, X_val)
    g.run(1e-5, Nit)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)
    g.print_hat('yhat_train vs y_train for Gradient', 'y_train', 'yhat_train', y_train, X_train)

    sd = SolveSteepDesc(y_train, X_train, y_val, X_val)
    sd.run(Nit)
    sd.print_result('Steepest Descent algorithm')
    sd.plot_err('Steepest Descent : square error', logy, logx)
    sd.print_hat('yhat_train vs y_train for Steepest Descent', 'y_train', 'yhat_train', y_train, X_train)

    st = SolveStoch(y_train, X_train, y_val, X_val)
    st.run(Nit, Nf, gamma)
    st.print_result('Stochastic gradient algorithm')
    st.plot_err('Stochastic gradient : square error', logy, logx)
    st.print_hat('yhat_train vs y_train for Stochastic', 'y_train', 'yhat_train', y_train, X_train)

    conj = SolveConj(y_train, X_train, y_val, X_val)
    conj.run()
    conj.print_result('Conjugate')
    conj.plot_err('Conjugate : square error', logy, logx)
    conj.print_hat('yhat_train vs y_train for Conjugate', 'y_train', 'yhat_train', y_train, X_train)

    ridge = SolveRidge(y_train, X_train, y_val, X_val)
    ridge.run(lamb)
    ridge.print_result('Ridge')
    best_lamb = ridge.run(lamb)

    m.plot_w('w:LLS')
    g.plot_w('w:Grad')
    sd.plot_w('w:Steep')
    st.plot_w('w:Stoch')
    conj.plot_w('w:Conjugate')
    ridge.plot_w('w:Ridge')


















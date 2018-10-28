import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lab0.sub.minimization import *

# std = (z - zmean) / deviation

if __name__ == "__main__":
    x = pd.read_csv("parkinsons_updrs.data")
    x.info()
    x.describe()
    x.plot()
    realdata = x.values  # if there is () = method while without it 's an attribute
    #np.random.shuffle(realdata)

    ##################################
    # shuffle is for the final version
    ##################################

    print("Matrix inside the file:\n", realdata)
    print("shape: ", np.shape(realdata))

    data = realdata[:, 4:22]
    print("Useful data :\n", data)
    Np, Nf = np.shape(data)
    print("shape: patients ", Np, "features ", Nf)

    data_train = data[0:int(Np/2), :]
    data_val = data[int(Np/2):int(Np*0.75), :]
    data_test = data[int(Np*0.75):Np, :]
    print("fine")

    mean = np.mean(data_train, 0)
    std = np.std(data_train, 0)

    data_train_norm = (data_train - mean)/std
    data_val_norm = (data_val - mean)/std
    data_test_norm = (data_test - mean)/std

    F0 = 7
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
    gamma = 1e-3
    N = 2
    lamb = 0.3

    m = SolveLLS(y_train, X_train)
    m.run()
    m.print_result('LLS')

    g = SolveGrad(y_train, X_train)
    g.run(1e-5, Nit)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)

    sd = SolveSteepDesc(y_train, X_train)
    sd.run(Nit)
    sd.print_result('Steepest Descent algorithm')
    sd.plot_err('Steepest Descent : square error', logy, logx)


    st = SolveStoch(y_train, X_train)
    st.run(Nit, Nf, gamma)
    st.print_result('Stochastic gradient algorithm')
    st.plot_err('Stochastic gradient : square error', logy, logx)

    mb = SolveMini(y_train, X_train)
    mb.run(Nit, N, gamma)
    mb.print_result('Minibach')
    mb.plot_err('Minibach : square error', logy, logx)

    conj = SolveConj(y_train, X_train)
    conj.run()
    conj.print_result('Conjugate')
    conj.plot_err('Conjugate : square error', logy, logx)

    ridge = SolveRidge(y_train, X_train)
    ridge.run(lamb)
    ridge.print_result('Ridge')
    ridge.plot_w('Ridge : square error')

    m.plot_w('LLS')
    g.plot_w('w:Grad')
    sd.plot_w('w:Steep')
    st.plot_w('w:Stoch')
    mb.plot_w('w:MiniB')
    conj.plot_w('w:Conjugate')
    ridge.plot_w('w:Ridge')


















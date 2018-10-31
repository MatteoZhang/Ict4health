import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lab0.sub.min import *

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
    Nit = 500
    gamma = 1e-5
    N = 2
    lamb = 0.3

    m = SolveLLS(y_train, X_train, y_val, X_val)
    m.run()
    m.print_result('LLS')

    g = SolveGrad(y_train, X_train, y_val, X_val)
    g.run(gamma, Nit)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)
    g.print_hat('hat')

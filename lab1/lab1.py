import pandas as pd
import numpy as np
from sub.min import *

if __name__ == "__main__":
    np.random.seed(2)  # random seed grant us an comparable results obtained in each algorithm
    x = pd.read_csv("parkinsons_updrs.csv")
    x.info()  # the panda framework display an easier to read matrix
    realdata = x.values  # convert the values in the file into a matrix specifically an Ndarray
    np.random.shuffle(realdata)  # the real data is shuffled using numpy


    data = realdata[:, 4:22]  # specifications of the lab : neglect the first 4 columns
    Np, Nf = np.shape(data)  # Np(# of rows) is the number of patients and Nf(# of columns) is the number of features

    data_train = data[0:int(Np/2), :]  # the training data is 50% of the whole data set
    data_val = data[int(Np/2):int(Np*0.75), :]  # validation data set 25%
    data_test = data[int(Np*0.75):Np, :]  # test data set 25%

    mean = np.mean(data_train, 0)  # returns a row of means
    std = np.std(data_train, 0)  # returns a row of standard deviations

    # standardizing our data means that an eventual offset will be not considered
    data_train_norm = (data_train - mean)/std
    data_val_norm = (data_val - mean)/std
    data_test_norm = (data_test - mean)/std

    # checking if the mean and standard deviation are respected
    mean_check = np.mean(data_train_norm, 0)  # returns a row of means
    std_check = np.std(data_train_norm, 0)  # returns a row of standard deviations

    F0 = 1  # F0 is the feature we want to choose in order to be the regressand in our case  it is the Total UPDRS
    # regressands y and regressors X set up
    y_train = data_train_norm[:, F0]
    X_train = np.delete(data_train_norm, F0, 1)
    y_val = data_val_norm[:, F0]
    X_val = np.delete(data_val_norm, F0, 1)
    y_test = data_test_norm[:, F0]
    X_test = np.delete(data_test_norm, F0, 1)


    # with the slicing operation we have to keep in mind to check the resulting shape
    # print(np.shape(y_train)) to check
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # some inizialization before starting the algorithms
    logx = 0
    logy = 1
    Nit = 300
    gamma = 1e-5
    mse_train = np.zeros((6,1),dtype=float)
    mse_val = np.zeros((6,1),dtype=float)
    mse_test = np.zeros((6,1),dtype=float)

    # these objects will be explained in the min file
    m = SolveLLS(y_train, X_train, y_val, X_val, y_test, X_test)
    mse_train[0], mse_val[0],mse_test[0]=m.run()
    m.print_result('LLS')
    m.print_hat('yhat_train vs y_train for LLs','yhat_train','y_train', y_train,X_train, mean[F0],std[F0])
    m.print_hat('yhat_test vs y_test for LLs', 'yhat_test', 'y_test', y_test, X_test, mean[F0],std[F0])

    g = SolveGrad(y_train, X_train, y_val, X_val, y_test, X_test)
    mse_train[1], mse_val[1], mse_test[1]=g.run(1e-5, Nit)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)
    g.print_hat('yhat_train vs y_train for Gradient', 'yhat_train', 'y_train', y_train, X_train, mean[F0],std[F0])
    g.print_hat('yhat_test vs y_test for Gradient', 'yhat_test', 'y_test', y_test, X_test, mean[F0],std[F0])

    sd = SolveSteepDesc(y_train, X_train, y_val, X_val, y_test, X_test)
    mse_train[2], mse_val[2], mse_test[2] =sd.run(Nit)
    sd.print_result('Steepest Descent algorithm')
    sd.plot_err('Steepest Descent : square error', logy, logx)
    sd.print_hat('yhat_train vs y_train for Steepest Descent', 'yhat_train','y_train',y_train, X_train,mean[F0],std[F0])
    sd.print_hat('yhat_test vs y_test for Steepest Descent', 'yhat_test', 'y_test', y_test, X_test, mean[F0],std[F0])

    st = SolveStoch(y_train, X_train, y_val, X_val, y_test, X_test)
    mse_train[3], mse_val[3], mse_test[3] =st.run(Nit, Nf, gamma)
    st.print_result('Stochastic gradient algorithm')
    st.plot_err('Stochastic gradient : square error', logy, logx)
    st.print_hat('yhat_train vs y_train for Stochastic', 'yhat_train', 'y_train', y_train, X_train, mean[F0],std[F0])
    st.print_hat('yhat_test vs y_test for Stochastic', 'yhat_test', 'y_test', y_test, X_test, mean[F0],std[F0])

    conj = SolveConj(y_train, X_train, y_val, X_val, y_test, X_test)
    mse_train[4], mse_val[4], mse_test[4] =conj.run()
    conj.print_result('Conjugate')
    conj.plot_err('Conjugate : square error', logy, logx)
    conj.print_hat('yhat_train vs y_train for Conjugate', 'yhat_train', 'y_train', y_train, X_train, mean[F0],std[F0])
    conj.print_hat('yhat_test vs y_test for Conjugate', 'yhat_test', 'y_test', y_test, X_test, mean[F0],std[F0])

    ridge = SolveRidge(y_train, X_train, y_val, X_val, y_test, X_test)
    mse_train[5], mse_val[5], mse_test[5] =ridge.run()
    ridge.print_result('Ridge')
    ridge.print_hat('yhat_train vs y_train for Ridge', 'yhat_train', 'y_train', y_train, X_train, mean[F0],std[F0])
    ridge.print_hat('yhat_test vs y_test for Ridge', 'yhat_test', 'y_test', y_test, X_test, mean[F0],std[F0])

    m.plot_w('w:LLS')
    g.plot_w('w:Grad')
    sd.plot_w('w:Steep')
    st.plot_w('w:Stoch')
    conj.plot_w('w:Conjugate')
    ridge.plot_w('w:Ridge')

    final_comparison = SolveMinProbl(y_train, X_train, y_val, X_val, y_test, X_test)
    final_comparison.print_mse(mse_train, mse_val, mse_test)




















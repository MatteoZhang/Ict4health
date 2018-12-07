import numpy as np
import matplotlib.pyplot as plt


class SolveMinProbl:
    def __init__(self, y, A):  # 111 and identity matrix
        self.matr = A  # matrix A allocating memory for parameters ST we don't need to call
        self.Np = y.shape[0]  # rows shape tell us the shape of the matrix
        self.Nf = A.shape[1]  # columns
        self.vect = y  # col vector
        self.sol = np.zeros((self.Nf, 1), dtype=float)  # column vector w solution
        self.min = 0.0
        self.err = 0
        return

    def plot_w(self, title='solution'):
        w = self.sol  # already initialized self.sol in the previous method
        n = np.arange(self.Nf)
        plt.figure()
        plt.plot(n, w)
        plt.xlabel('n')
        plt.ylabel('w(n)')
        plt.grid()
        plt.title(title)
        plt.show()
        return

    def print_result(self, title):
        print(title, ' :')
        print('the optimum weight vector is: ')
        print(self.sol)  # w
        return

    def plot_err(self, title='Square error', logy=1, logx=0):
        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return


class SolveRidge(SolveMinProbl):
    def run(self, lamb=0.5):
        A = self.matr
        y = self.vect
        I = np.eye(self.Nf)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + lamb * I), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w) - y)
        return

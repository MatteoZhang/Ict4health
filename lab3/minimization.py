import numpy as np
import matplotlib.pyplot as plt


class SolveMinProbl:
    def __init__(self, y, A):  # 111 and identity matrix
        self.matr = A  # matrix A allocating memory for parameters ST we don't need to call
        self.Np = y.shape[0]  # rows shape tell us the shape of the matrix
        self.Nf = A.shape[1]  # columns
        self.vect = y  # col vector
        self.sol = np.zeros((self.Nf, y.shape[1]), dtype=float)  # column vector w solution
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


class SolveRidge(SolveMinProbl):
    def run(self, lamb=10):
        A = self.matr
        y = self.vect
        I = np.eye(self.Nf)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+lamb*I), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w) - y)
        return w

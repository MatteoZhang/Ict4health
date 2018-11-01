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
    def plot_w(self,title = 'solution'):
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
    def print_result(self , title):
        print(title, ' :')
        print('the optimum weight vector is: ')
        print(self.sol)  # w
        return
    def plot_err(self, title='Square error', logy=1, logx=0):
        err = self.err
        plt.figure()
        if(logy == 0) & (logx == 0):
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
class SolveLLS(SolveMinProbl):  # this class belongs to SolveMinProbl
    def run(self):  # the inpu
        A = self.matr
        y = self.vect
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w)-y)
class SolveGrad(SolveMinProbl):
    def run(self, gamma, Nit):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        for it in range(Nit):
            grad = 2*np.dot(A.T, (np.dot(A, w)-y))
            w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w)-y)**2/self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveSteepDesc(SolveMinProbl):
    def run(self, Nit):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            H = 4 * np.dot(A.T, A)
            gamma2 = (np.linalg.norm(grad)**2)/np.dot(np.dot(grad.T, H), grad)
            w = w - gamma2 * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)**2/self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveStoch(SolveMinProbl):
    def run(self, Nit, gamma):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        row = np.zeros((1, self.Nf), dtype=float)
        for it in range(Nit):
            for i in range(self.Np):
                for j in range(self.Nf):
                    row[0, j] = A[i, j]
                grad = 2 * np.dot(row.T,(np.dot(row, w) - y[i]))
                w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)**2/self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveMini(SolveMinProbl):
    def run(self, Nit, N, gamma): #N is number of minibatches
        if (N > self.Np) | (self.Np % N != 0):
            print("function not valid")
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        m = int(self.Np/N)
        for it in range(Nit):
            for i in range(self.Nf):
                for j in range(N):
                    grad = 2 * np.dot(A[j:(j*m+m), :].T, (np.dot(A[j:(j*m+m), :], w) - y[j:(j*m+m)]))
                    w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)**2/self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveConj(SolveMinProbl):
    def run(self):
        self.err = np.zeros((self.Nf, 2), dtype=float)
        A = self.matr
        y = self.vect
        w = np.zeros((self.Nf, 1), dtype=float)
        it = 0
        Q = 2 * np.dot(A.T, A)
        b = 2 * np.dot(A.T, y)
        g = -b
        d = -g
        for it in range(self.Nf):
            #if np.dot(np.dot(d.T, Q), d) == 0:
            #   break
            a = -1 * np.dot(d.T, g)/np.dot(np.dot(d.T, Q), d)
            w = w + a * d
            g = g + a * np.dot(Q, d)
            beta = np.dot(np.dot(g.T, Q), d)/np.dot(np.dot(d.T, Q), d)
            d = -1 * g + beta*d
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)**2/self.Np
        self.sol = w
        self.min = self.err[it, 1]

class SolveRidge(SolveMinProbl):
    def run(self, lamb=0.5):
        A = self.matr
        y = self.vect
        I = np.eye(self.Nf)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+lamb*I), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w) - y)


import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    def __init__(self, y, A, y_val, A_val):  # initialization of the variable
        self.matr = A  # matrix and other allocations
        self.vect = y
        self.matr_val = A_val
        self.vect_val = y_val
        self.Np = A.shape[0]  # shape[0] is rows so y is a column vector
        self.Nf = A.shape[1]  # columns
        self.sol = np.zeros((self.Nf, 1), dtype=float)  # column vector w solution
        self.min = 0.0
        self.err = 0
        return
    def plot_w(self,title = 'solution'):
        w = self.sol  # already initialized self.sol in the previous method
        n = np.arange(self.Nf)  # number fo feature
        plt.figure()
        plt.plot(n, w)
        plt.xlabel('n')
        plt.ylabel('w(n)')

        #this is only for the lab1 not lab0
        plt.xticks(ticks=range(self.Nf),
                   labels=['motor_UPDRS', 'total_UPDRS', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5',
                           'Jitter:DDP', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
                           'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'], rotation='vertical')
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
            plt.plot(err[:, 0], err[:, 1], label='train')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], label='train')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1], label='train')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1], label='train')
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 2], label='val')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 2], label='val')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 2], label='val')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 2], label='val')
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.legend()
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return
    def print_hat(self, title, xlabel, ylabel, y, A):
        plt.figure()
        w = self.sol
        yhat = np.dot(A, w)
        plt.scatter(y, yhat)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.figure()
        plt.hist(yhat-y, 50)
        plt.xlabel('bins')
        plt.ylabel('yhat-y')
        plt.title('histogram related to: '+title)
        plt.grid()
        plt.show()

class SolveLLS(SolveMinProbl):  # this class belongs to SolveMinProbl
    def run(self):  # the input
        A = self.matr
        y = self.vect
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w)-y)
class SolveGrad(SolveMinProbl):
    def run(self, gamma, Nit):
        self.err = np.zeros((Nit, 3), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        for it in range(Nit):
            grad = 2*np.dot(A.T, (np.dot(A, w)-y))
            w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w)-y)**2/self.Np
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w)-y_val)**2/self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveSteepDesc(SolveMinProbl):
    def run(self, Nit):
        self.err = np.zeros((Nit, 3), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            H = 4 * np.dot(A.T, A)
            gamma2 = (np.linalg.norm(grad)**2)/np.dot(np.dot(grad.T, H), grad)
            w = w - gamma2 * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)**2/self.Np
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveStoch(SolveMinProbl):
    def run(self, Nit, Nf, gamma):
        self.err = np.zeros((Nit, 3), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        row = np.zeros((1, self.Nf), dtype=float)
        for it in range(Nit):
            for i in range(self.Np):
                for j in range(self.Nf):
                    row[0, j] = A[i, j]
                grad = 2 * row.T * (np.dot(row, w) - y[i])
                w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y)**2/self.Np
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveMini(SolveMinProbl):
    def run(self, Nit, N, gamma): #N is number of minibatches
        if (N > self.Np) | (self.Np % N != 0):
            print("function not valid")
        self.err = np.zeros((Nit, 3), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
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
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / self.Np
        self.sol = w
        self.min = self.err[it, 1]
class SolveConj(SolveMinProbl):
    def run(self):
        self.err = np.zeros((self.Nf, 3), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
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
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / self.Np
        self.sol = w
        self.min = self.err[it, 1]

class SolveRidge(SolveMinProbl):
    def run(self, lamb=0.5):
        A = self.matr
        y = self.vect
        I =np.eye(self.Nf)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+lamb*I), A.T), y)
        self.sol = w
        self.min = np.linalg.norm(np.dot(A, w) - y)
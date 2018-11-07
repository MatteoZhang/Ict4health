import numpy as np
import matplotlib.pyplot as plt

class SolveMinProbl:
    """
    This class is created to inizialize all the variable and it will provide methods to print the final results.

    Methods
    -------
    plot_w(self,title = 'solution')
        It plots the solution of our minimization problem which is a vector of Nf elements.
        w is our solution vector.
    print_result(self , title)
        Prints the results in the python console in order to have more accurate solution.
    plot_err(self, title='Square error', logy=1, logx=0)
        It plots the error in 4 different modes
    print_hat(self, title, xlabel, ylabel, y, A)
        Gives the plot of the actual regressand vs the regressand obtained from our algorithm
    """
    def __init__(self, y, A, y_val, A_val, y_test, A_test):  # initialization of the variable
        """
        Parameters
        ----------
        y: regressand of the training set
        A: regressor of the training set
        y_val: regressand of the validation set
        A_val: reggressor of the validation set

        """
        self.matr = A  # allocations
        self.vect = y
        self.matr_val = A_val
        self.vect_val = y_val
        self.matr_test = A_test
        self.vect_test = y_test
        self.Np = A.shape[0]  # shape[0] is rows and Np represent Number of Patients
        self.Nf = A.shape[1]  # columns Nf represent Number of Features
        self.sol = np.zeros((self.Nf, 1), dtype=float)  # column vector w solution
        self.err = 0

    def plot_w(self, title='solution'):
        """
        This method uses the matplotlib in order to make plots of the w vector : the solution vector

        Parameters
        ----------
        title: str
            title of the solution vector w
        """
        w = self.sol  # already initialized self.sol in the previous method
        n = np.arange(self.Nf)  # number fo feature
        plt.figure()
        plt.plot(n, w)
        plt.xlabel('n')
        plt.ylabel('w(n)')
        plt.xticks(ticks=range(self.Nf),
                   labels=['motor_UPDRS', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5',
                           'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA',
                           'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'], rotation='vertical')
        plt.grid()
        plt.title(title)
        plt.show()
    def print_result(self , title):
        """
        Console results in order to have accurate numbers

        Parameters
        ----------
        title: str
            title of the results displayed in the console
        """
        print(title, ' :')
        print('the optimum weight vector is: ')
        print(self.sol)  # w
    def plot_err(self, title='Square error', logy=1, logx=0):
        """
        Parameters
        ----------

        title: str
        logy: int
            0 if linear scale on y axis , the vertical one, 1 if log scale in y axis
        logx: int
            0 if linear on x axis , 1 if log scale on x axis
        """
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
        plt.title(title+' (mse)')
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
    def print_hat(self, title, xlabel, ylabel, y, A, mean , std):
        """
        Paramenters
        -----------
        title: str
        xlabel: str
            label for the x axis
        ylabel: str
            lab for the y axis
        y: array
            the values taken from the dataset
        A: matrix
            the values taken from the dataset
        mean: float
            scalar of the feature we want to regress on
        std: float
            scalar of the feature we want to regress on
        """
        plt.figure()
        w = self.sol
        yhat = (np.dot(A, w) * std)+mean
        y = (y * std)+mean
        plt.plot(np.linspace(0, 60), np.linspace(0, 60), 'orange')
        plt.scatter(yhat, y, s=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.figure()
        plt.hist(y-yhat, 50)
        plt.xlabel('bins')
        plt.ylabel('y-yhat')
        plt.title('histogram related to: '+title)
        plt.grid()
        plt.show()
    def print_mse(self, mse_train,mse_val,mse_test,std):
        """
        Parameters
        ----------
        mse_train : array
        mse_val : array
        mse_test : array
        """
        plt.figure()
        plt.title("Comparison of the means quare errors of the last interaction\n of each Algorithm")
        plt.plot(np.arange(len(mse_train)), mse_train*std, label='train')
        plt.plot(np.arange(len(mse_val)), mse_val*std, label='validation')
        plt.plot(np.arange(len(mse_test)), mse_test*std, label='test')
        plt.ylabel('error of the last interaction')
        plt.xticks(ticks=range(6),
                   labels=['LLS','Gradient','Steepest Descent','Stochastic','Conjugate','Ridge'], rotation='vertical')
        plt.grid()
        plt.legend()
        plt.show()
class SolveLLS(SolveMinProbl):  # this class belongs to SolveMinProbl
    """ Linear Least Square """
    def run(self):  # the input
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        self.err = np.zeros((1, 4), dtype=float)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol = w
        self.err[0, 1] = np.linalg.norm(np.dot(A, w)-y)**2/A.shape[0]
        self.err[0, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / A_val.shape[0]
        self.err[0, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        return self.err[0, 1], self.err[0, 2], self.err[0, 3]
class SolveGrad(SolveMinProbl):
    """ Gradient Algorithm """
    def run(self, gamma, Nit):
        self.err = np.zeros((Nit, 4), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        for it in range(Nit):
            grad = 2*np.dot(A.T, (np.dot(A, w)-y))
            w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w)-y)**2/A.shape[0]
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w)-y_val)**2/A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]
class SolveSteepDesc(SolveMinProbl):
    """ Solve Steepest Descend Algorithm """
    def run(self, Nit):
        self.err = np.zeros((Nit, 4), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        for it in range(Nit):
            grad = 2 * np.dot(A.T, (np.dot(A, w) - y))
            H = 4 * np.dot(A.T, A)
            gamma2 = (np.linalg.norm(grad)**2)/np.dot(np.dot(grad.T, H), grad)
            w = w - gamma2 * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y) ** 2 / A.shape[0]
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 /A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]
class SolveStoch(SolveMinProbl):
    """ Stochastic Algorithm """
    def run(self, Nit, Nf, gamma):
        self.err = np.zeros((Nit, 4), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
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
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y) ** 2 / A.shape[0]
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]
class SolveMini(SolveMinProbl):
    """" MiniBatch Algorithm """
    def run(self, Nit, N, gamma): #N is number of minibatches
        if (N > self.Np) | (self.Np % N != 0):
            print("function not valid")
        self.err = np.zeros((Nit, 4), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        w = np.random.rand(self.Nf, 1)  # uniform pdf random var range 0,1
        it = 0
        m = int(self.Np/N)
        for it in range(Nit):
            for i in range(self.Nf):
                for j in range(N):
                    grad = 2 * np.dot(A[j:(j*m+m), :].T, (np.dot(A[j:(j*m+m), :], w) - y[j:(j*m+m)]))
                    w = w - gamma * grad
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y) ** 2 / A.shape
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]
class SolveConj(SolveMinProbl):
    """" Conjugate Algorithm """
    def run(self):
        self.err = np.zeros((self.Nf, 4), dtype=float)
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
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
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y) ** 2 / A.shape[0]
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]
class SolveRidge(SolveMinProbl):
    """" Ridge Algorithm """
    def run(self):
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        I = np.eye(self.Nf)
        stop_lamb = 100
        self.err = np.zeros((stop_lamb, 4), dtype=float)
        for it in range(stop_lamb):
            w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+float(it)*I), A.T), y)
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(np.dot(A, w) - y) ** 2 / A.shape[0]
            self.err[it, 2] = np.linalg.norm(np.dot(A_val, w) - y_val) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(np.dot(A_test, w) - y_test) ** 2 / A_test.shape[0]
        best_lamb = np.argmin(self.err[:, 2])
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + best_lamb * I), A.T), y)
        self.sol = w
        err = self.err
        print("best lambda is :", best_lamb)

        #plotting the figure of the ridge respect to the lambda which is from 0-1
        plt.figure()
        plt.semilogy(err[:, 0], err[:, 1], label='train')
        plt.semilogy(err[:, 0], err[:, 2], label='val')
        plt.xlabel('lambda')
        plt.ylabel('e(lambda)')
        plt.legend()
        plt.title('Ridge error respect to lambda')
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]

from sub.minimization import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.random.seed(7)
    plt.close('all')
    Np = 100  # row
    Nf = 9  # col
    A = np.random.randn(Np, Nf)  # gaussian random var
    y = np.random.randn(Np, 1)

    Nit = 5000
    gamma = 1e-4
    logx = 0
    logy = 1
    N = Np

    m = SolveLLS(y, A)
    m.run()
    m.print_result('LLS')
    m.plot_w('LLS')

    g = SolveGrad(y, A)
    g.run(gamma, Nit)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)

    sd = SolveSteepDesc(y, A)
    sd.run(Nit)
    sd.print_result('Steepest Descent algorithm')
    sd.plot_err('Steepest Descent : square error', logy, logx)

    st = SolveStoch(y, A)
    st.run(Nit, gamma)
    st.print_result('Stochastic gradient algorithm')
    st.plot_err('Stochastic gradient : square error', logy, logx)

    mb = SolveMini(y, A)
    mb.run(Nit, N, gamma)
    mb.print_result('Minibach')
    mb.plot_err('Minibach : square error', logy, logx)

    conj = SolveConj(y, A)
    conj.run()
    conj.print_result('Conjugate')
    conj.plot_err('Conjugate : square error', logy, logx)

    rd = SolveRidge(y, A)
    rd.run()
    rd.print_result('Ridge')
    rd.plot_w('Ridge')

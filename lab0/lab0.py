from sub.minimization import *
import numpy as np

if __name__ == "__main__":
    np.random.seed(7)
    plt.close('all')
    Np = 4  # row
    Nf = 4  # col
    A = np.random.randn(Np, Nf)  # gaussian random var
    y = np.random.randn(Np, 1)
    m = SolveLLS(y, A)
    m.run()
    m.print_result('LLS')
    m.plot_w('LLS')
    Nit = 500
    gamma = 1e-3
    logx = 0
    logy = 0
    N = 2

    g = SolveGrad(y, A)
    g.run(gamma, Nit * 10)
    g.print_result('Gradient algorithm')
    g.plot_err('Gradient algorithm : square error', logy, logx)

    sd = SolveSteepDesc(y, A)
    sd.run(Nit)
    sd.print_result('Steepest Descent algorithm')
    sd.plot_err('Steepest Descent : square error', logy, logx)

    st = SolveStoch(y, A)
    st.run(Nit, Nf, gamma)
    st.print_result('Stochastic gradient algorithm')
    st.plot_err('Stochastic gradient : square error', logy, logx)

    mb = SolveMini(y, A)
    mb.run(Nit, N, gamma)
    mb.print_result('Minibach')
    mb.plot_err('Minibach : square error', logy, logx)

    conj = SolveConj(y, A)
    conj.run(Nit)
    conj.print_result('Conjugate')
    conj.plot_err('Conjugate : square error', logy, logx)

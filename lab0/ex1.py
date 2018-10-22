import numpy as np
import matplotlib.pyplot as mpl

Nr = 4  # Number of rows Np
Nc = 4  # Number of columns Nf
A = np.random.randn(Nr, Nc)  # gaussian random variable with Nr rows and Nc col
w_id = np.random.randn(Nc)  # gaussian
y = np.dot(A, w_id)

AAT = np.dot(A, A.T)
AATinv = np.linalg.inv(AAT)
AATinvAT = np.dot(AATinv, A.T)
v = np.dot(AATinvAT, y)

print('y is :', y)
print('A is :', A)

w = np.dot(np.linalg.pinv(A), y)  # linear lest square sudo inverse in linalg only in rectangular
print('vector w is: ', w)

mpl.figure()
mpl.plot(w)
mpl.xlabel('n')
mpl.ylabel('w(n)')
mpl.grid()
mpl.title('solution')
mpl.show()






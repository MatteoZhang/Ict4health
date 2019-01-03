import numpy as np

if __name__ == '__main__':
    mask5 = np.zeros((5, 5))
    for i in range(5):
        mask5[2, i] = 1
        mask5[i, 2] = 1
    mask3 = np.zeros((3, 3))
    for i in range(3):
        mask5[1, i] = 1
        mask5[i, 1] = 1

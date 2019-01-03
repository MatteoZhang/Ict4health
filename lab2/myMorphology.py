import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    action='ignore', module='matplotlib.figure', category=UserWarning,
    message=('This figure includes Axes that are not compatible with tight_layout, '
             'so results might be incorrect.')
)


def fill_holes(toro):
    row, col = toro.shape
    tmp = np.copy(toro)
    if tmp[int(np.round(row/2)),int(np.round(col/2))] == 0:
        tmp[int(np.round(row/2)),int(np.round(col/2))] = 1
    toro = tmp
    return toro


def erode(to_erode):
    row, col = to_erode.shape
    tmp = np.copy(to_erode)
    for i in range(1,row-1):
        for j in range(1,col-1):
            if tmp[i,j] == 1:
                cnt = 0
                try:
                    if tmp[i-1,j] == 0:
                        cnt += 1
                    if tmp[i,j+1] == 0:
                        cnt += 1
                    if tmp[i+1,j] == 0:
                        cnt += 1
                    if tmp[i,j-1] == 0:
                        cnt += 1
                    if cnt >= 3:
                        tmp[i,j] = 0
                except IndexError:
                    pass
    to_erode = tmp
    return to_erode


def dilate(to_dilate):
    row, col = to_dilate.shape
    tmp = np.copy(to_dilate)
    for i in range(1,row-1):
        for j in range(1,col-1):
            if tmp[i,j] == 0:
                cnt = 0
                try:
                    if tmp[i-1,j] == 1:
                        cnt += 1
                    if tmp[i,j+1] == 1:
                        cnt += 1
                    if tmp[i+1,j] == 1:
                        cnt += 1
                    if tmp[i,j-1] == 1:
                        cnt += 1
                    if cnt >= 3:
                        tmp[i,j] = 1
                except IndexError:
                    pass
    to_dilate = tmp
    return to_dilate


if __name__ == '__main__':
    A = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                   [0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
                   [0, 1, 0, 0, 0, 1, 1, 1, 1, 0],
                   [0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
                   [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    B = dilate(A)
    plt.matshow(A)
    plt.matshow(B)
    plt.show()


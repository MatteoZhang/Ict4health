import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(
    action='ignore', module='matplotlib.figure', category=UserWarning,
    message=('This figure includes Axes that are not compatible with tight_layout, '
             'so results might be incorrect.')
)


def connected_label(matrix):
    label = np.copy(matrix)
    col, row = matrix.shape
    k = 0
    for i in range(col):
        for j in range(row):
            if matrix[i, j] == 1 and matrix[i, j-1] == 0:
                k += 1
            if label[i, j] == 1:
                label[i, j] = k
    print("label matrix: \n", label)
    dictionary = {}
    k = 0
    for i in range(col):
        for j in range(row):
            if label[i, j] > 0:
                if label[i, j] not in dictionary.values():
                    k += 1
                    dictionary[k] = label[i, j]
                    if label[i - 1, j] > 0:
                        dictionary[k] = label[i - 1, j]
                        label[i, j] = label[i - 1, j]
                    if label[i - 1, j - 1] > 0:
                        dictionary[k] = label[i - 1, j - 1]
                        label[i, j] = label[i - 1, j - 1]
                    if label[i - 1, j + 1] > 0:
                        dictionary[k] = label[i - 1, j + 1]
                        label[i, j] = label[i - 1, j + 1]
                    if label[i, j - 1] > 0:
                        dictionary[k] = label[i, j - 1]
                        label[i, j] = label[i, j - 1]
    k = 0
    for i in range(col):
        for j in range(row):
            if label[i, j] > 0:
                k += 1
                if label[i + 1, j] > 0:
                    dictionary[k] = label[i+1, j]
                    label[i, j] = label[i + 1, j]
    print("\n", label, "\n")
    print(dictionary)
    return label


def max_label(label_matrix):
    return


def fill_holes(toro):
    col, row  = toro.shape
    tmp = np.copy(toro)

    return toro


def erode(to_erode):
    col, row  = to_erode.shape
    tmp = np.copy(to_erode)

    to_erode = tmp
    return to_erode


def dilate(to_dilate):
    col, row = to_dilate.shape
    tmp = np.copy(to_dilate)

    to_dilate = tmp
    return to_dilate


def floodfill(matrix, x, y):
    return matrix


def polish_dots(withdots):
    col, row = withdots.shape
    for i in range(col):
        for j in range(row):
            withdots[0, j] = 0
            withdots[i, 0] = 0
            withdots[col-1, j] = 0
            withdots[i, row-1] = 0
    for i in range(col):
         for j in range(row):
            if i > 0 and i < col-1 and j > 0 and j < row-1:
                if withdots[i, j] == 1:
                    if withdots[i-1, j] == 0 and withdots[i, j-1] == 0 and withdots[i+1, j] == 0 and withdots[i, j+1] == 0:
                        withdots[i, j] = 0
                if withdots[i, j] == 0:
                    if withdots[i-1, j] == 1 and withdots[i, j-1] == 1 and withdots[i+1, j] == 1 and withdots[i, j+1] == 1:
                        withdots[i, j] = 1
    return withdots


if __name__ == '__main__':
    A = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                   [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    A[0][0] = 0
    print(A)
    print(A.shape)
    plt.matshow(A)

    B = polish_dots(A)
    plt.matshow(B)

    C = connected_label(B)
    plt.matshow(C)

    plt.show()


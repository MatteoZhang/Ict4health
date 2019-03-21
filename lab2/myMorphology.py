import numpy as np

class Polish(object):

    def connected_label(self, matrix):
        label = np.copy(matrix)
        col, row = matrix.shape
        k = 0
        matrix[0,0] = 0
        for i in range(col):
            for j in range(row):
                if matrix[i, j] == 1 and matrix[i, j - 1] == 0:
                    k += 1
                if label[i, j] == 1:
                    label[i, j] = k
        k = 0
        for i in range(2, col-1):
            for j in range(2, row-1):
                if label[i, j] > 0:
                    if label[i-1, j] > 0:
                        label[label == label[i, j]] = label[i-1, j]
                    if label[i-1, j-1] > 0:
                        label[label == label[i, j]] = label[i-1, j-1]
                    if label[i-1, j+1] > 0:
                        label[label == label[i, j]] = label[i-1, j+1]
        return label

    def max_label(self, label_matrix):
        col, row = label_matrix.shape
        dictionary = {}

        for i in range(col):
            for j in range(row):
                if label_matrix[i, j] > 0:
                    dictionary[label_matrix[i, j]] = 0

        for i in range(col):
            for j in range(row):
                if label_matrix[i, j] > 0:
                    for k in range(len(dictionary)):
                        if label_matrix[i, j] == list(dictionary.keys())[k]:
                            dictionary[label_matrix[i, j]] += 1

        tmp = max(dictionary, key=dictionary.get)
        for i in range(col):
            for j in range(row):
                if label_matrix[i, j] > 0:
                    if label_matrix[i, j] != tmp:
                        label_matrix[i, j] = 0
        label_matrix[label_matrix > 0] = 1
        return label_matrix

    def dilate(self,to_dilate):
        label = np.copy(to_dilate)
        col, row = to_dilate.shape
        # dilate
        for i in range(2, col-1):
            for j in range(2, row-1):
                if label[i, j] == 1:
                    to_dilate[i - 1, j] = 1
                    to_dilate[i + 1, j] = 1
                    to_dilate[i, j - 1] = 1
                    to_dilate[i, j + 1] = 1
                    to_dilate[i - 1, j - 1] = 1
                    to_dilate[i - 1, j + 1] = 1
                    to_dilate[i + 1, j - 1] = 1
                    to_dilate[i + 1, j + 1] = 1
        return to_dilate

    def erode(self, to_erode):
        label = np.copy(to_erode)
        col, row = to_erode.shape
        # erode
        for i in range(2, col - 1):
            for j in range(2, row - 1):
                if (label[i - 1, j] == 1 and label[i + 1, j] == 1 and
                        label[i, j - 1] == 1 and label[i, j + 1] == 1):
                    to_erode[i, j] = 2
        return to_erode

    def area(self, image):
        col, row = image.shape
        area = 0
        for i in range(col):
            for j in range(row):
                if image[i, j] == 2:
                    area += 1
        return area

    def perimeter(self, image):
        col, row = image.shape
        perimeter = 0
        for i in range(col):
            for j in range(row):
                if image[i, j] == 1:
                    perimeter += 1
        return perimeter

    def only_perimeter(self, image):
        image[image == 2] = 0
        return image



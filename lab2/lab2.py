"""
@author: MZ
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings(
    action='ignore', module='matplotlib.figure', category=UserWarning,
    message=('This figure includes Axes that are not compatible with tight_layout, '
             'so results might be incorrect.')
)


class Figure(object):
    def __init__(self, file):
        self.file = file
        self.original = mpimg.imread(self.file)
        self.N1, self.N2, self.N3 = self.original.shape
        N1, N2, N3 = self.N1, self.N2, self.N3
        self.im_2d = self.original.reshape((N1 * N2, N3))
        # pixel in position i.j goes to position k=(i-1)*N2+j)
        self.Nr, self.Nc = self.im_2d.shape
        self.N_cluster = 4
        self.k_means = KMeans(n_clusters=self.N_cluster, random_state=0)
        self.k_means.fit(self.im_2d)
        self.k_means_centroids = self.k_means.cluster_centers_.astype('uint8')

    def show_figure(self, title='title'):
        plt.figure()
        plt.imshow(self.original, interpolation=None)
        plt.title(title)
        plt.pause(0.05)

    def quantized(self):
        im_2d_quantized = self.im_2d.copy()
        for kc in range(self.N_cluster):
            quant_color_kc = self.k_means_centroids[kc, :]
            ind = (self.k_means.labels_ == kc)
            im_2d_quantized[ind, :] = quant_color_kc
        return im_2d_quantized.reshape((self.N1, self.N2, self.N3))


class Polish(object):

    def connected_label(self, matrix):
        label = np.copy(matrix)
        col, row = matrix.shape
        k = 0
        for i in range(col):
            for j in range(row):
                if matrix[i, j] == 1 and matrix[i, j - 1] == 0:
                    k += 1
                if label[i, j] == 1:
                    label[i, j] = k
        k = 0
        for i in range(col):
            for j in range(row):
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
        return label_matrix

    def fill_hole(self, to_fill):
        inverse = 1 - to_fill
        col, row = to_fill.shape
        for i in range(2, col-1):
            for j in range(2, row-1):
                if to_fill[i, j] == 1:
                    if inverse[i-1, j] == 1:
                        to_fill[i-1, j] = 1
                    if inverse[i+1, j] == 1:
                        to_fill[i+1, j] = 1
                    if inverse[i, j-1] == 1:
                        to_fill[i, j-1] = 1
                    if inverse[i, j+1] == 1:
                        to_fill[i, j+1] = 1
        return to_fill

    def area(self):
        return

    def perimeter(self):
        return


class Circle(object):
    def __init__(self, area_obj):
        self.area = area_obj

    def area(self):
        return

    def perimeter(self):
        return


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    plt.close('all')
    filein = 'moles/melanoma_4.jpg'
    # filein = 'moles/low_risk_4.jpg'
    # filein = 'moles/medium_risk_4.jpg'
    fig_original = Figure(filein)
    fig_original.show_figure('original image')
    fig_quantized = fig_original.quantized()
    plt.figure()
    plt.title('quantized image')
    plt.imshow(fig_quantized, interpolation=None)
    plt.pause(0.05)
    # 1: find the darkest color found by k-means darkest color corresponds to the mole
    centroids = fig_original.k_means_centroids
    sc = np.sum(centroids, axis=1)
    i_col = sc.argmin()
    im_clust = fig_original.k_means.labels_.reshape(fig_original.N1, fig_original.N2)
    # plt.matshow(im_clust)
    # 3: find the positions i,j where im_clust is equal to i_col
    # the 2D Ndarray zpos stores the coordinates i,j only of the pixels
    # in cluster i_col
    zpos = np.argwhere(im_clust == i_col)
    # 4: ask the user to write the number of objects belonging to
    # cluster i_col in the image with quantized colors

    N_spots_str = input("How many distinct dark spots can you see in the image? ")
    N_spots = int(N_spots_str)
    # 5: find the center of the mole
    if N_spots == 1:
        center_mole = np.median(zpos, axis=0).astype(int)
    else:
        # use K-means to get the N_spots clusters of zpos
        kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
        kmeans2.fit(zpos)
        centers = kmeans2.cluster_centers_.astype(int)
        # the mole is in the middle of the picture:
        center_image = np.array([fig_original.N1 // 2, fig_original.N2 // 2])
        center_image.shape = (1, 2)
        d = np.zeros((N_spots, 1))
        for k in range(N_spots):
            d[k] = np.linalg.norm(center_image - centers[k, :])
        center_mole = centers[d.argmin(), :]
    # 6: take a subset of the image that includes the mole
    c0 = center_mole[0]
    c1 = center_mole[1]
    RR, CC = im_clust.shape
    stepmax = min([c0, RR - c0, c1, CC - c1])
    cond = True
    area_old = 0
    surf_old = 1
    step = 10  # each time the algorithm increases the area by 2*step pixels
    # horizontally and vertically
    im_sel = (im_clust == i_col)  # im_sel is a boolean NDarray with N1 rows and N2 columns
    im_sel = im_sel * 1  # im_sel is now an integer NDarray with N1 rows and N2 columns
    while cond:
        subset = im_sel[c0 - step:c0 + step + 1, c1 - step:c1 + step + 1]
        area = np.sum(subset)
        Delta = np.size(subset) - surf_old
        surf_old = np.size(subset)
        if area > area_old + 0.01 * Delta:
            step = step + 10
            area_old = area
            cond = True
            if step > stepmax:
                cond = False
        else:
            cond = False
            # subset is the serach area
    plt.matshow(subset)
    plt.title('search area')

    polishing = Polish()
    polish = polishing.connected_label(subset)
    plt.matshow(polish)
    plt.title('label matrix')
    polished = polishing.max_label(polish)
    plt.matshow(polished)
    plt.title('biggest connected component')

    filled = polishing.fill_hole(polished)
    plt.matshow(filled)
    plt.title('hole filling process')

    #find areas and perimeter and ratios

    plt.tight_layout()
    plt.show()

# TODO  complete code



"""
@author: MZ
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage.morphology import binary_fill_holes


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
    def __init__(self, subset_matrix):
        self.subset = subset_matrix
        self.n_row, self.n_col = self.subset.shape
        self.polished = np.zeros((self.n_row, self.n_col), dtype=float)

    def polish(self):
        n_row = self.n_row
        n_col = self.n_col
        print("---polishing matrix with dimension %sx%s---" % (n_row, n_col))
        self.polished = binary_fill_holes(self.subset).astype(int)
        plt.matshow(self.polished)
        plt.title("filled holes")

    def area(self):
        return

    def perimeter(self):
        return


class Circle(object):
    def __init__(self, area_circle):
        self.area = area_circle

    def perimeter(self):
        return


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    plt.close('all')
    # filein = 'moles/melanoma_1.jpg'
    # filein = 'moles/low_risk_1.jpg'
    filein = 'moles/medium_risk_1.jpg'
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

    polished = Polish(subset)
    polished.polish()
    #area_polished = polished.area()
    #perimeter_polished = polished.peremeter()
    #circle = Circle(area_polished)
    #perimeter_circle = circle.perimeter()
    #ratio = perimeter_polished/perimeter_circle
    #print("Area: %s \nPerimeter: %s \nRatio: %s" % (area_polished, perimeter_polished, ratio))

    plt.show()

# TODO  complete code



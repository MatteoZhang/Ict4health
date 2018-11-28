import numpy as np
import matplotlib.image as mpimg
import  matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.set_printoptions(precision=2)
plt.close('all')
filein = 'moles/medium_risk_1.jpg'
im_original = mpimg.imread(filein)
plt.figure()
plt.imshow(im_original)
plt.title('original image')
#plt.draw()
plt.pause(0.1)

N1, N2, N3 = im_original.shape
print("image has %s x %s x %s shape" % (N1, N2, N3))
im_2D = im_original.reshape((N1 * N2, N3))
Nr, Nc = im_2D.shape

Ncluster = 3  # quantization levels
kmeans = KMeans(n_clusters=Ncluster, random_state=0)
kmeans.fit(im_2D)

kmeans_centroids=kmeans.cluster_centers_.astype('uint8')
im_2D_quant=im_2D.copy()

for kc in range(Ncluster):
    quant_color_kc = kmeans_centroids[kc, :]
    ind = (kmeans.labels_ == kc)
    im_2D_quant[ind, :] = quant_color_kc
im_quant = im_2D_quant.reshape((N1, N2, N3))
plt.figure()
plt.imshow(im_quant, interpolation=None)
plt.title('image with quantized colors')
#plt.draw()
plt.pause(0.1)

centroids = kmeans_centroids
sc = np.sum(centroids, axis=1)
i_col = sc.argmin()
im_clust = kmeans.labels_.reshape(N1, N2)
plt.matshow(im_clust)
zpos = np.argwhere(im_clust == i_col)

N_spots = int(input('How many spots can u see in the image? '))
if N_spots == 1:
    center_mole = np.median(zpos, axis=0).astype(int)
else:
    kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
    kmeans2.fit(zpos)
    centers = kmeans2.cluster_centers_.astype(int)
    center_image = np.array([N1 // 2, N2 // 2])
    center_image.shape = (1, 2)
    d = np.zeros((N_spots, 1))
    for k in range(N_spots):
        d[k] = np.linalg.norm(center_image - centers[k, :])
    center_mole = centers[d.argmin(), :]

cond = True
area_old = 0
step = 10
c0 = center_mole[0]
c1 = center_mole[1]
im_sel = (im_clust == i_col)
im_sel = im_sel * 1
while cond:
    subset = im_sel[c0 - step:c0 + step + 1, c1 - step:c1 + step + 1]
    area = np.sum(subset)
    if area > area_old:
        step = step + 10
        area_old = area
        cond = True
    else:
        cond = False




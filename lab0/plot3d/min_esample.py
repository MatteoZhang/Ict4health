import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

for tighten in False,True:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([1,2], [1,2], [1,2])
    if tighten:
        fig.tight_layout()
plt.show()

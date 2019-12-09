import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from functionUtil import *
fig = plt.figure()
ax = plt.axes(projection="3d")

x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)

x1, x2 = np.meshgrid(x1, x2)
y = np.zeros((50, 50))

functions = [islo_uni_F1, islo_uni_F2, islo_uni_F3, islo_uni_F4, islo_uni_F5, islo_uni_F6, islo_uni_F7, islo_uni_F8,
             islo_multi_F9, islo_multi_F10, islo_multi_F11, islo_multi_F12, islo_multi_F13, islo_multi_F14, islo_multi_F15, islo_multi_F16,
             islo_hybrid_F17, islo_hybrid_F18, islo_hybrid_F19, islo_hybrid_F20, islo_hybrid_F21, islo_hybrid_F22, islo_hybrid_F23,
             islo_compos_F24, islo_compos_F25, islo_compos_F26, islo_compos_F27, islo_compos_F28, islo_compos_F29, islo_compos_F30]

for function in functions:
    try:
        for i in range(50):
            for j in range(50):
                solution = np.array([x1[i, j], x2[i, j]])
                y_ij = function(solution=solution)
                y[i, j] += y_ij
        surf = ax.plot_surface(x1, x2, y, cmap='jet', edgecolor='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(function.__name__+".png", dpi=100)
        plt.close()
    except Exception as e:
        print(e)
        pass
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
# print(x1)
# print(x2)

# y_plot = (np.abs(x1_plot)+np.abs(x2_plot)) +(np.abs(x1_plot)*np.abs(x2_plot))
# y = 2*x1**2 + 3*x2**2
# y = x1**2 + 10e6*x2**2
# y = -20 * np.exp(-0.2 * (1/2*(x1**2+x2**2)) ** 0.5) - \
#            np.exp(1/2*(np.cos(2 * np.pi * x1)+np.cos(2 * np.pi * x1))) + 20 + np.e

# y = 1 - np.cos(2*np.pi*(x1**2+x2**2)**0.5) + 0.1*(x1**2+x2**2)**0.5
# y = (100*(x1**2 - x2)**2 + (x1 - 1)**2)
# y = -20 * np.exp(-0.2 * (1/2*((5.12*x1/100)**2+(5.12*x2/100)**2)) ** 0.5) - \
#            np.exp(1/2*(np.cos(2 * np.pi * (5.12*x1/100))+np.cos(2 * np.pi * (5.12*x2/100)))) + 20 + np.e
surf = ax.plot_surface(x1, x2, y, cmap='jet', edgecolor='none')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
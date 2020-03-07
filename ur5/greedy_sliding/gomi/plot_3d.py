from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def func1(x, y):
    return x**2 + -y**2 + 3*x

x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)

X, Y = np.meshgrid(x, y)
Z = func1(X, Y)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_wireframe(X, Y, Z)
plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def func1(x, y):
    return x**2 + -y**2 + 3*x

x = np.linspace(-0.325, 0.48, 100)
y = np.arange(-0.075, 0.075, 100)
z = np.sin(x-0.325) + 1.3
print z
# X, Y = np.meshgrid(x, y)
# Z = func1(X, Y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x,y,z)
# # ax.plot_wireframe(X, Y, Z)
# plt.show()
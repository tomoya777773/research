from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random


def func1(x, y):
    return 0.01 * np.sin(2*np.pi/(0.015704*4)*(x - (0.4 - 0.15704/2))) + 1.3

# x = np.linspace(0.4 - 0.15704/2, 0.4 + 0.15704/2,50)
# y = np.linspace(-0.075, 0.075, 50)

x = np.linspace(0.33, 0.47,40)
y = np.linspace(-0.07, 0.07, 40)


# num = 50
# x = [random.uniform(0.4 - 0.15704/2, 0.4 + 0.15704/2) for i in range(num)]
# y = [random.uniform(-0.075, 0.075) for i in range(num)]

print func1(0.44, 1)

# x = np.linspace(-0.325, 0.48, 100)
# y = np.linspace(-0.075, 0.075, 100)

X, Y = np.meshgrid(x, y)
Z = func1(X, Y)
# print X.flatten()
# print Y.flatten().shape

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()
sin_list = []
for i in range(len(X.flatten())):
    sin_list.append([X[i], Y[i], Z[i]])
sin_list = np.array(sin_list)
print sin_list
# print sin_list[:, 0]
np.save("../data1/judge_point_list.npy", sin_list)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# ax.set_xlim(0.4 - 0.15704/2, 0.4 + 0.15704/2)
# ax.set_ylim(-0.075, 0.075)
# ax.set_zlim(1.28, 1.32)

ax.scatter(sin_list[:, 0], sin_list[:, 1], sin_list[:, 2])
# ax.plot_wireframe(X, Y, Z)
plt.show()
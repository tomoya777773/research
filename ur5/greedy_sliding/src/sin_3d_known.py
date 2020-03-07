from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random


def func1(x):
    return 0.01 * np.sin(2*np.pi/(0.015704*4)*(x - (0.4 - 0.15704/2))) + 1.3

# x = np.linspace(0.4 - 0.15704/2, 0.4 + 0.15704/2, 50)
# y = np.linspace(-0.075, 0.075, 50)

num = 5000
x = [random.uniform(0.4 - 0.15704/2, 0.4 + 0.15704/2) for i in range(num)]
y = [random.uniform(-0.075, 0.075) for i in range(num)]
z = map(lambda x : func1(x), x)

sin_list = []
for i in range(num):
    sin_list.append([x[i], y[i], z[i]])
print np.array(sin_list)


np.save("../data1/surf_sin_known_5000.npy", sin_list)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# ax.set_xlim(0.4 - 0.15704/2, 0.4 + 0.15704/2)
# ax.set_ylim(-0.075, 0.075)
# ax.set_zlim(1.28, 1.32)

ax.scatter(x,y,z)
plt.show()
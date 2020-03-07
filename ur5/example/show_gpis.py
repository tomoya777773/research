import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

po = np.load("mean_zero_500.npy")
X = po[0]
Y = po[1]
Z = po[2]


# x=[]
# y=[]
# z=[]
#
# for i in range(len(X_)):
#     for j in range(3):
#         if j == 0:
#             x.append(X_[i][j])
#         elif j == 1:
#             y.append(X_[i][j])
#         elif j == 2:
#             z.append(X_[i][j])
#
# X, Y , Z = np.meshgrid(x, y, z)
# print X



fig = plt.figure()
# ax = Axes3D(fig)
ax = fig.gca(projection='3d')

# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")
# ax.plot(x,y, z, "o", color="#00aa00", ms=4, mew=1)

ax.plot_surface(X, Y, Z,cmap=plt.cm.viridis)
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm)
# ax.plot_wireframe(x,y,z)

plt.show()

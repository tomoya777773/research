import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

X = np.load("X1.npy")
Y = np.load("Y1.npy").T[0]

print X
print Y

X_ = []
for i in range(len(Y)):
    if Y[i] == 0:
        X_.append(X[i])
print X_
x=[]
y=[]
z=[]

for i in range(len(X_)):
    for j in range(3):
        if j == 0:
            x.append(X_[i][j])
        elif j == 1:
            y.append(X_[i][j])
        elif j == 2:
            z.append(X_[i][j])

fig = pyplot.figure()
ax = Axes3D(fig)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.plot(x, y, z, "o", color="#00aa00", ms=4, mew=1)
# ax.plot_wireframe(x,y,z)

pyplot.show()

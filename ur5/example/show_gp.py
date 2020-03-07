import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import GPy

kernel = GPy.kern.RBF(2, ARD=True)

data = np.load("gp_data.npy")

x1 = np.delete(data[0][0], -1)
x2 = np.delete(data[1], -1)
y = data[2][0][0]


input = np.hstack((np.reshape(x1,(-1, 1)), np.reshape(x2, (-1, 1))))
output = np.reshape(y, (-1, 1))


print input
def meanfunc(x):
    return x
model = GPy.models.GPRegression(input, output, kernel, mean_function=ConstMean())
model.optimize(messages=False, max_iters=1e5)

x = np.linspace(-1.1, 1.1, 100)
y = np.linspace(1.1, -1.1, 100)

## prediction
# x_pred = np.array([x, y]).T
# y_pred = model.predict(x_pred)[0]


X, Y = np.meshgrid(x, y)

Z = np.zeros([100,100])

for i in range(len(x)):
    for j in range(len(y)):
        input_pred = np.array( [[ x[i], y[j] ]] )
        Z[i][j]= model.predict(input_pred)[0]
print Z
plt.pcolormesh(X, Y, Z)
plt.colorbar()
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(X, Y, Z, c="bisque")

plt.show()
#
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(model.X[:,0],model.X[:,1],model.Y.ravel(), c="r")
#
#
# # ax.scatter(X,Y,Z)
# # ax.scatter(model.X[:,0],model.X[:,1],model.Y.ravel(), c="r")
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# # plt.imshow(Z)
#
# # plt.colorbar()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# plt.colorbar()
plt.show()

# plt.pcolormesh(X, Y, Z)
# plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x1, x2, y)
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.show()

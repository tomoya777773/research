import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from scipy.stats import norm, multivariate_normal




def func(x, y):
    m=2
    mean = np.zeros(m)
    sigma = np.eye(m) * 30
    X = np.c_[np.ravel(x), np.ravel(y)] * 30

    Y_plot = 100 * (multivariate_normal.pdf(x=X, mean=mean, cov=sigma) - 0.35*multivariate_normal.pdf(x=X, mean=mean, cov=sigma*0.5))-0.045

    Y_plot = Y_plot.reshape(x.shape)

    return Y_plot

# x = np.linspace(-0.3, 0.3, 50) # x goes from -3 to 3, with 256 steps
# y = np.linspace(-0.3, 0.3, 50) # y goes from -3 to 3, with 256 steps
# XX, YY = np.meshgrid(x, y) # combine all x with all y

# ZZ = func(XX, YY)

# print ZZ.min()
# import ipdb; ipdb.set_trace()

theta, phi = np.linspace(-np.pi, np.pi, 40), np.linspace(0, np.pi/4, 40)
# theta, phi = np.linspace(np.pi/8, np.pi/8*7, 1), np.linspace(np.pi/8, np.pi/9*4, 1)

r = 0.4
THETA, PHI = np.meshgrid(theta, phi)
R = np.cos(PHI) * r*2
XX = R * np.sin(PHI) * np.cos(THETA)
YY = R * np.sin(PHI) * np.sin(THETA)
ZZ = func(XX, YY)

print ZZ.min()
# print func(1,1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(np.ravel(XX), np.ravel(YY), np.ravel(ZZ))
ax.set_title("Scatter Plot")

# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)

plt.show()

X = np.ravel(XX)[:,None]
Y = np.ravel(YY)[:,None]
Z = np.ravel(ZZ)[:,None]


print ZZ.shape

sphere_po = np.concatenate([X,Y,Z], 1)
sphere_po_2d = np.array([XX, YY, ZZ])

print sphere_po.shape
print sphere_po_2d.shape

np.save("../data/object/object1", sphere_po)
np.save("../data/object/object1_2d", sphere_po_2d)

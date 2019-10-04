import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

theta, phi = np.linspace(-np.pi, np.pi, 20), np.linspace(0, np.pi/4, 10)
# theta, phi = np.linspace(np.pi/8, np.pi/8*7, 1), np.linspace(np.pi/8, np.pi/9*4, 1)

r = 0.21
THETA, PHI = np.meshgrid(theta, phi)
R = np.cos(PHI) * r*2
XX = R * np.sin(PHI) * np.cos(THETA) * 2
YY = R * np.sin(PHI) * np.sin(THETA) * 2
ZZ = R * np.cos(PHI)  - r


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(np.ravel(XX), np.ravel(YY), np.ravel(ZZ))
ax.set_title("Scatter Plot")

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()

X = np.ravel(XX)[:,None]
Y = np.ravel(YY)[:,None]
Z = np.ravel(ZZ)[:,None]

sphere_po = np.concatenate([X,Y,Z], 1)
sphere_po_2d = np.array([XX, YY, ZZ])

print sphere_po.shape
print sphere_po_2d.shape

np.save("../data/ellipse/ellipse_po_21", sphere_po)
np.save("../data/ellipse/ellipse_po_21_2d", sphere_po_2d)

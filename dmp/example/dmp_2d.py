import sys
sys.path.append("../")

from dmp_lib import DMPs_discrete
import numpy as np
import matplotlib.pyplot as plt


# dmp_x = DMPs_discrete(dmps=1, bfs=10, w=np.zeros((1,10)))
# dmp_y = DMPs_discrete(dmps=1, bfs=10, w=np.zeros((1,10)))

# x_track,dx_track,ddx_track = dmp_x.rollout()
# y_track,dy_track,ddy_track = dmp_y.rollout()

x = np.linspace(0, 8, 100)
y = np.sin(x)

# path_x = np.linspace(0, 6, 100)
# path_y = np.linspace(0.8, 1, 100)


dmp = DMPs_discrete(dmps=2, bfs=100)

dmp.imitate_path(y_des=np.array([x, y]))
print dmp.goal
dmp.goal[0] = 10
dmp.goal[1] = 2
y_track,dy_track,ddy_track = dmp.rollout()


plt.plot(y_track[:, 0], y_track[:, 1], lw = 2)
plt.plot(x, y, lw = 2)
plt.xlim(0, 15)
plt.ylim(-1, 1.5)
plt.show()
import sys
sys.path.append("../")

from dmp_lib import DmpsGpis
from gpis import GaussianProcessImplicitSurface

import numpy as np
import matplotlib.pyplot as plt

beta = 20.0 / np.pi
gamma = 100
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])
# num_obstacles = 5
# obstacles = np.random.random((num_obstacles, 2))*2 - 1
obstacles = np.array([5,0.3])

def contact_judge(current_position):
    return round(current_position[1] - np.sin(current_position[0]), 3) == 0

def prevent_insert(judge_position):
    return judge_position[1] > np.sin(judge_position[0])

def avoid_obstacles(y, dy, goal):
    p = np.zeros(2)

    for obstacle in obstacles:
        # based on (Hoffmann, 2009)

        # if we're moving
        if np.linalg.norm(dy) > 1e-5:

            # get the angle we're heading in
            phi_dy = -np.arctan2(dy[1], dy[0])
            R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],
                             [np.sin(phi_dy), np.cos(phi_dy)]])
            # calculate vector to object relative to body
            obj_vec = obstacle - y
            # rotate it by the direction we're going
            obj_vec = np.dot(R_dy, obj_vec)
            # calculate the angle of obj relative to the direction we're going
            phi = np.arctan2(obj_vec[1], obj_vec[0])
            # print phi, phi_dy
            dphi = gamma * phi * np.exp(-beta * abs(phi))
            R = np.dot(R_halfpi, np.outer(obstacle - y, dy))
            pval = np.nan_to_num(np.dot(R, dy) * dphi)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval
    return p

# Create GPIS model
x = np.arange(0, 8, 0.1)
y = np.sin(x)
position_data = np.column_stack([x,y])
label_data = np.zeros(len(x)).reshape(-1, 1)
gpis = GaussianProcessImplicitSurface(position_data, label_data)


# Create DMP model
path_x = np.linspace(3, 7, 100)
path_y = 0.2 * np.sin(path_x) + 0.5
# path_y = np.ones(len(path_x)) - 0.8
dmp = DmpsGpis(dmps=2, bfs=100)
dmp.imitate_path(y_des=np.array([path_x, path_y]))
# dmp.goal[0] = 9
# dmp.goal[1] = 0.6569865987187891


"""test"""
current_position = np.array([path_x[0], path_y[0]])
path = np.zeros((len(path_x), 2))

for i in range(len(path_x)):
    n,d = gpis.direction_func(current_position)

    if contact_judge(current_position):
        direction = n * 100
        # print "dddddddddddd:", -d
    else:
        direction = -n * 100
        # print "nnnnnnnnnnnn:", -n

    print dmp.y
    y_track,dy_track,ddy_track = dmp.rollout(direction=direction)
    # print y_track[-1]

    if i == 0:
        path[i] = y_track[i]
        current_position = y_track[i]
    else:
        judge = True
        interval = 100
        dt = (y_track[i] - path[i-1]) / interval
        n = 1
        halfway = path[i-1]
        while n < interval:
            judge = prevent_insert(halfway)
            if not judge:
                break
            halfway = path[i-1] + dt * n
            n += 1
            # print judge
            # print n

        path[i] = halfway
        current_position = halfway

# print path

plt.plot(path[:, 0], path[:, 1], lw = 2)
plt.plot(x, y, lw = 2)
plt.plot(path_x, path_y)
plt.xlim(0, 8)
plt.ylim(-1, 1.1)
plt.show()
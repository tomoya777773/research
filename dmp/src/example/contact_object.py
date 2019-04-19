
import numpy as np
import matplotlib.pyplot as plt

import pydmps.dmp_discrete

beta = 20.0 / np.pi
gamma = 100
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])

num_obstacles = 100
x_ = np.linspace(0, 8, num_obstacles)
y_ = np.sin(x_)
obstacles = np.column_stack([x_,y_])



# print obstacles
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

            pval = -np.nan_to_num(np.dot(R, dy) * dphi)

            # check to see if the distance to the obstacle is further than
            # the distance to the target, if it is, ignore the obstacle
            if np.linalg.norm(obj_vec) > np.linalg.norm(goal - y):
                pval = 0

            p += pval
    return p


# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=100,w=np.zeros((2,100)))
y_track = np.zeros((dmp.timesteps, dmp.n_dmps))
dy_track = np.zeros((dmp.timesteps, dmp.n_dmps))
ddy_track = np.zeros((dmp.timesteps, dmp.n_dmps))
goal = np.array([1,1])
dmp.goal = goal
# dmp.y0 = np.array([1,1])
dmp.reset_state()

for t in range(dmp.timesteps):
    y_track[t], dy_track[t], ddy_track[t] = \
            dmp.step(external_force=avoid_obstacles(dmp.y, dmp.dy, goal))

plt.figure(1, figsize=(6,6))
plot_goal, = plt.plot(dmp.goal[0], dmp.goal[1], 'gx', mew=3)
for obstacle in obstacles:
    plot_obs, = plt.plot(obstacle[0], obstacle[1], 'rx', mew=3)
plot_path, = plt.plot(y_track[:,0], y_track[:, 1], 'b', lw=2)
plt.title('DMP system - obstacle avoidance')
plt.axis('equal')
plt.xlim([-1.1,3])
plt.ylim([-1.1,1.1])
plt.show()


import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib import DmpsGpis
from gpis import GaussianProcessImplicitSurface


beta = 2
gamma = 100
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])

num_obstacles = 100
x_ = np.linspace(0, 8, num_obstacles)
y_ = np.sin(x_)
obstacles = np.column_stack([x_,y_])
# print obstacles


def contact_judge(current_position):
    return round(current_position[1] - np.sin(current_position[0]), 2) == 0

def prevent_insert(judge_position):
    return judge_position[1] > np.sin(judge_position[0])

def avoid_obstacles(dy, direction, goal):
    p = np.zeros(2)
    if np.linalg.norm(dy) > 1e-5:

        # get the angle we're heading in
        phi_dy = -np.arctan2(dy[1], dy[0])
        # print "dffsagsa", phi_dy*180/np.pi
        R_dy = np.array([[np.cos(phi_dy), -np.sin(phi_dy)],
                            [np.sin(phi_dy), np.cos(phi_dy)]])

        d_vec = np.dot(R_dy, direction)    # rotate it by the direction we're going
        phi = np.arctan2(d_vec[1], d_vec[0])  # calculate the angle of direction(n or d) to the direction we're going

        R_phi = np.array([[np.cos(phi), -np.sin(phi)],
                          [np.sin(phi), np.cos(phi)]])


        dphi = gamma * phi * np.exp(-beta / abs(phi))
        # print "phipppppppppppp", phi*180/np.pi

        print "phi_dy:", phi_dy
        print "vec:", d_vec
        print "phi", phi

        # if phi > 0:
        #     R_half =  R_halfpi
        # else:
        #     R_half = -R_halfpi
        R = np.dot(R_halfpi, np.outer(direction, dy))


        pval = np.nan_to_num(np.dot(R_halfpi, dy) * dphi)

        p += pval
    # print "pppp", p
    return p

if __name__ == '__main__':

    # Create GPIS model
    x = np.arange(1, 8, 0.1)
    y = np.sin(x)
    position_data = np.column_stack([x,y])
    label_data = np.zeros(len(x)).reshape(-1, 1)
    gpis = GaussianProcessImplicitSurface(position_data, label_data, a=1)

    # Create DMP model
    path_x = np.linspace(3, 6, 100)
    # path_y = 0.1 * np.sin(path_x) + 0.8183676841431136
    path_y = 0.1 * np.sin(path_x) + 0.2

    dmp = DmpsGpis(dmps=2, bfs=100)
    dmp.imitate_path(y_des=np.array([path_x, path_y]))


    """test"""
    current_position = np.array([path_x[0], path_y[0]])
    dy = [0, 0]
    y_track = []
    cnt = 1
    for i in range(dmp.timesteps):
        print "-----------------------------------"
        print "count:", cnt
        print "position:",current_position
        print "dy:", dy
        n,d = gpis.direction_func(current_position)

        if contact_judge(current_position):
            direction = d
            print "dddddddddddd:", d
            # print "nnnnnnnnnnnn:", -n.T[0]

        else:
            direction = -n.T[0]
            print "nnnnnnnnnnnn:", -n.T[0]

        external_force=avoid_obstacles(dy, direction, dmp.goal)
        print "eeeeeeeeeeeeeeeeeeeeeee", external_force
        y, dy, ddy = dmp.step(external_force=external_force)

        judge = True
        interval = 100
        dt = (y - current_position) / interval
        n = 0
        while n < interval:
            judge = prevent_insert(current_position)
            # print judge
            if not judge:
                # current_position -= dt
                break
            current_position += dt

            n += 1

        y = current_position
        dmp.y = current_position
        y_track.append(np.copy(y))

        cnt += 1


    y_track = np.array(y_track)




    # print y_track[:, 0]
    # print path
    # print path[-1]
    plt.plot(y_track[:, 0], y_track[:, 1], lw = 2)
    # plt.plot(x, y, lw = 2)
    plt.plot(path_x, path_y)
    plt.plot(x_, y_, c ="r")
    plt.xlim(0, 8)
    plt.ylim(-4, 4)
    # plt.savefig("sliding1.png")
    plt.show()
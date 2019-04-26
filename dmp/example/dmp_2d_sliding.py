import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib import DmpsGpis
from gpis import GaussianProcessImplicitSurface
import math

beta = 2
gamma = 200
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])


def contact_judge(current_position):
    return round(current_position[1] - np.sin(current_position[0]), 1) == 0

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
    x_ = np.arange(0, 8, 0.1)
    y_ = np.sin(x_)
    position_data = np.column_stack([x_,y_])
    label_data = np.zeros(len(x_)).reshape(-1, 1)
    gpis = GaussianProcessImplicitSurface(position_data, label_data, a=1)

    # Create DMP model
    path_x = np.linspace(3, 6, 100)
    # path_y = 0.1 * np.sin(path_x) + 0.8183676841431136
    # path_y = 0.1*np.sin(path_x) + 0.18
    path_y = np.full(len(path_x), 0.3)

    # x1 = np.arange(3, 4.6, 0.1)
    # y1 = np.full(len(x1), 0.4)

    # x2, y2 =[],[]

    # for _x in np.linspace(180,-180,60):
    #     x2.append(0.5*math.sin(math.radians(_x))+4.5)
    #     y2.append(0.5*math.cos(math.radians(_x))+0.9)

    # x3 = np.arange(4.5, 6, 0.1)
    # y3 = np.full(len(x3), 0.4)

    # path_x = np.append(x1,x3)
    # path_y = np.append(y1, y3)


    # for i in range(len(path_x)):
    #     plt.scatter(path_x[i], path_y[i])
    #     plt.pause(0.02)



    dmp = DmpsGpis(dmps=2, bfs=100)
    dmp.imitate_path(y_des=np.array([path_x, path_y]))


    """test"""
    current_position = np.array([path_x[0], path_y[0]])
    dy = [0, 0]
    y_track = []
    cnt = 1

    while round(current_position[0] - dmp.goal[0], 1) != 0:
        if cnt > 3000: break

        print "-----------------------------------"
        print "count:", cnt
        print "position:",current_position
        print "dy:", dy
        n,d = gpis.direction_func(current_position)
        print "-------------------", contact_judge(current_position)
        print "nnnnnnnnnnnnnnnn", n
        if contact_judge(current_position):
            direction = n
            print "nnnnnnnnn:", n

        else:
            direction = -n
            # print "normal:", -n

        external_force=avoid_obstacles(dy, direction, dmp.goal)
        print "external_force:", external_force

        y, dy, ddy = dmp.step(external_force=external_force)

        judge = True
        interval = 100
        dt = (y - current_position) / interval
        n = 0

        if prevent_insert(current_position):

            while n < interval:
                judge = prevent_insert(current_position)
                # print judge
                if not judge:
                    current_position -= dt
                    break
                current_position += dt

                n += 1
        # plt.scatter(current_position[0], current_position[1])
        # plt.pause(0.001)

        # dmp.y = current_position
        y_track.append(np.copy(current_position))
        # print "goal:", dmp.goal
        cnt += 1

    y_track = np.array(y_track)

    plt.plot(y_track[:, 0], y_track[:, 1], lw = 2)
    plt.plot(path_x, path_y)
    plt.plot(x_, y_, c ="r")
    plt.xlim(0, 8)
    plt.ylim(-4, 4)
    # plt.savefig("sliding1.png")
    plt.show()
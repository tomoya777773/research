import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from dmp_lib import DmpsGpis
from gpis import GaussianProcessImplicitSurface
import math

beta = 1.5
gamma = 300
R_halfpi = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                     [np.sin(np.pi / 2.0), np.cos(np.pi / 2.0)]])


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

        # print "phi:", phi*180/np.pi
        print "phi_dy:", phi_dy
        print "vec:", d_vec
        print "phi", phi

        p = np.nan_to_num(np.dot(R_halfpi, dy) * dphi)

    return p

if __name__ == '__main__':

    # Create GPIS model
    x_ = np.arange(1, 12, 0.1)
    y_ = np.sin(x_) - 0.02
    position_data = np.column_stack([x_,y_])
    label_data = np.zeros(len(x_)).reshape(-1, 1)
    gpis = GaussianProcessImplicitSurface(position_data, label_data, a=1)

    # Create DMP model
    path_x = np.linspace(2, 10, 100)
    path_y = np.full(len(path_x), 1.2)

    # path_y = 0.1 * np.sin(path_x) + 0.8183676841431136
    # path_y = 0.1*np.sin(path_x) + 1.0

    # for i in range(len(path_x)):
    #     plt.scatter(path_x[i], path_y[i])
    #     plt.pause(0.02)

    dmp = DmpsGpis(dmps=2, bfs=500, dt= 0.005)
    dmp.imitate_path(y_des=np.array([path_x, path_y]))


    """test"""
    current_position = np.array([path_x[0], path_y[0]])
    dy = np.array([0, 0])
    y_track = []
    cnt = 1

    while abs(current_position[0] - dmp.goal[0]) > 0.2:
        if cnt > 1000: break

        print "-----------------------------------"
        print "count:", cnt
        print "position:",current_position
        print "dy:", dy
        print "-------------------", contact_judge(current_position)

        n,d = gpis.direction_func(current_position)

        if contact_judge(current_position):
            direction = n
            d_judge = True
            # print "nnnnnnnnn:", n

        else:
            direction = -n
            d_judge = False
            # print "normal:", -n

        external_force=avoid_obstacles(dy, direction, dmp.goal)
        print "external_force:", external_force

        y, dy, ddy = dmp.step(external_force=external_force*5)
        dy = dy/np.linalg.norm(dy)

        judge = True
        interval = 100
        dt = (y - current_position) / interval
        n = 0


        if prevent_insert(current_position):

            while n < interval:
                current_position += dt

                # if d_judge:
                #     if not contact_judge(current_position):
                #         break

                if not prevent_insert(current_position):
                    current_position -= dt
                    break
                n += 1



            y_track.append(np.copy(current_position))
        cnt += 1

        plt.scatter(current_position[0], current_position[1])
        plt.pause(0.001)

    y_track = np.array(y_track)

    plt.plot(y_track[:, 0], y_track[:, 1], lw = 2)
    plt.plot(path_x, path_y)
    plt.plot(x_, y_, c ="r")
    plt.xlim(0, 11)
    plt.ylim(-3, 3)
    # plt.savefig("sliding1.png")
    plt.show()




    # x1 = np.arange(3, 5, 0.1)
    # y1 = np.full(len(x1), 0.4)

    # x2 = np.arange(5, 4, 0.1)
    # y2 = np.full(len(x2), 0.4)


    # # x2, y2 =[],[]

    # # for _x in np.linspace(180,-180,60):
    # #     x2.append(0.5*math.sin(math.radians(_x))+4.5)
    # #     y2.append(0.5*math.cos(math.radians(_x))+0.9)

    # x3 = np.arange(4, 6, 0.1)
    # y3 = np.full(len(x3), 0.4)

    # path_x = np.append(np.append(x1,x2), x3)
    # path_y = np.append(np.append(y1, y2), y3)


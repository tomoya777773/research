from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import greedy_step_normal as gs
from tqdm import tqdm

la = np.load("../data1/la_list_known.npy")
sin_list = np.load("../data1/judge_point_list.npy")
sin_list_ = sin_list.tolist()
# print la.shape
# print np.amin(sin_list[:,0])

for i in range(len(la)):
    sin_list_[i].append(la[i][0])
    sin_list_[i].append(la[i][1])

sin_list = np.asarray(sin_list_)

sin_list = sin_list[np.where((sin_list[:,0] > 0.33) & (sin_list[:,0] < 0.47)\
           & (sin_list[:,1] > -0.07) & (sin_list[:,1] < 0.07))]

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(sin_list[:,0], sin_list[:,1], sin_list[:,2], c='r', alpha=0.8, label = "correct",s=70)
# # ax.scatter(x,y,z)
# plt.show()

true_list, false_list = [], []
count, true_count, false_count = 0,0,0
n = len(sin_list)
print n
for i in tqdm(range(n)):
    if sin_list[i][2] > 1.3:
        if sin_list[i][3] < 0 and sin_list[i][4] < 0:
            true_list.append(sin_list[i])
            true_count += 1
        else:
            false_list.append(sin_list[i])
            false_count += 1

    else:
        if sin_list[i][3] < 0 and sin_list[i][4] < 0:
            false_list.append(sin_list[i])
            false_count += 1
        else:
            true_list.append(sin_list[i])
            true_count += 1

    count += 1
    # print("count:", count)

print "true count:", true_count
print "false count:", false_count
print "true accurency:", float(true_count)/n
print "false accurency:", float(false_count)/n

true_list = np.array(true_list)
false_list = np.array(false_list)


plt.scatter(true_list[:,0], true_list[:,1], c="r", s=50)
plt.scatter(false_list[:,0], false_list[:,1], c="b", s=50)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")

# ax.surface(true_list[:,0], true_list[:,1], true_list[:,2], c='r', alpha=0.8, label = "correct",s=50)
# ax.scatter(false_list[:,0], false_list[:,1], false_list[:,2], c='b', alpha=0.8, label = "incorrect",s=50)
plt.show()
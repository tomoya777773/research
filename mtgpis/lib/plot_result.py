import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")

name = "deco"
var_list        = np.load("../data/{}/mtgpis/value/var.npy".format(name))
simirality_list = np.load("../data/{}/mtgpis/value/simirality.npy".format(name))
surf_list       = np.load("../data/{}/mtgpis/value/surf.npy".format(name))

var_list_gp     = np.load("../data/{}/gpis/value/var.npy".format(name))
surf_list_gp    = np.load("../data/{}/gpis/value/surf.npy".format(name))

# print(var_list)
# print(simirality_list)
# print(surf_list)

# step = len(var_list)
x = np.arange(len(var_list)) * 0.01

simirality_t12 = []
for simirality in simirality_list:
    simirality_t12.append(simirality[1][0] / simirality[1][1])

fig = plt.figure(figsize=(12.0, 3.0))

ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(x, var_list, c="blue")
ax1.plot(x, var_list_gp, c="red")
ax1.set_xlabel("Travel length [m]")
ax1.set_ylabel("Uncertainty measure")

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(x, surf_list, c="blue")
ax2.plot(x, surf_list_gp, c="red")
ax2.set_ylim([0, 1.1])
ax2.set_xlabel("Travel length [m]")
ax2.set_ylabel("Surface error")

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(x, simirality_t12, c="blue")
ax3.set_ylim([0, 1.1])
ax3.set_xlabel("Travel length [m]")
ax3.set_ylabel("Simirality measure")

plt.tight_layout()
plt.show()

# var_list        = np.load("../data/low_low/mtgpis/value/var.npy")
# simirality_list = np.load("../data/low_low/mtgpis/value/simirality.npy")
# surf_list       = np.load("../data/low_low/mtgpis/value/surf.npy")
# # print(var_list)
# # print(simirality_list)
# # print(surf_list)

# step = len(var_list)

# simirality_t12 = []
# for simirality in simirality_list:
#     simirality_t12.append(simirality[1][0] / simirality[1][1])

# fig = plt.figure(figsize=(18.0, 6.0))

# ax1 = fig.add_subplot(1, 3, 1)
# ax1.plot(range(step), var_list)

# ax2 = fig.add_subplot(1, 3, 2)
# ax2.plot(range(step), surf_list)

# ax3 = fig.add_subplot(1, 3, 3)
# ax3.plot(range(step), simirality_t12)
# plt.show()
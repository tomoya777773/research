import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
import math

import matplotlib.font_manager as fon
del fon.weight_dict['roman']
fon._rebuild()

def set_plt():
    plt.xlim(0, 0.6)
    plt.ylim(0, 0.6)
    # plt.xticks(color="None")
    # plt.yticks(color="None")
    # plt.tick_params(length=0)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    plt.axes().set_aspect('equal', 'datalim')


# 全体
plt.rcParams['figure.figsize']    = (5,5)
plt.rcParams['figure.dpi']        = 300
plt.rcParams['font.size']         = 22
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.sans-serif'] = 'helvetica'
# plt.rcParams['font.sans-serif'] = ['Arial']

# 軸
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
plt.rcParams['xtick.labelsize']   = 15 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize']   = 15 # 軸だけ変更されます
plt.rcParams['xtick.major.width'] = 1.2 # x目盛の太さ
plt.rcParams['ytick.major.width'] = 1.2 # y目盛の太さ
plt.rcParams['xtick.major.size']  = 5 # x軸目盛りの長さ
plt.rcParams['ytick.major.size']  = 5 # y軸目盛りの長さ
plt.rcParams['axes.linewidth']    = 1.5 #軸の太さを設定。目盛りは変わらない
# plt.rcParams['axes.grid']         = True
# plt.rcParams['grid.linestyle']    = '--'
# plt.rcParams['grid.linewidth']    = 0.3
# plt.figure(figsize=(4, 4), dpi=300)

# 凡例
# plt.rcParams["legend.markerscale"]   = 1
# plt.rcParams["legend.fancybox"]      = False # 丸角
# plt.rcParams["legend.framealpha"]    = 1 # 透明度の指定、0で塗りつぶしなし
# plt.rcParams["legend.edgecolor"]     = 'black' # edgeの色を変更
# plt.rcParams["legend.handlelength"]  = 1.5 # 凡例の線の長さを調節
# plt.rcParams["legend.labelspacing"]  = 0.4 # 垂直方向（縦）の距離の各凡例の距離
# plt.rcParams["legend.handletextpad"] = 0.5 # 凡例の線と文字の距離の長さ



fig = plt.figure(figsize=(5, 5), dpi=300)
ax = fig.add_subplot(111)


rec1 = pat.Rectangle(xy = (0.15, 0.15), width = 0.3, height = 0.3,
angle = 0, color = "darkred", alpha = 1, label='External shape')

rec2 = pat.Rectangle(xy = (0.17, 0.17), width = 0.26, height = 0.26,
angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

set_plt()

ax.add_patch(rec1)
ax.add_patch(rec2)
# ax.add_patch(rec3)
# ax.add_patch(rec4)

ax.annotate('Outer shape', family='helvetica',
            xy=(0.3, 0.45), xycoords='data', size=18,
            xytext=(-10, 40), textcoords='offset points',
            bbox=dict(boxstyle="round", fc='white', edgecolor='gray'),
            arrowprops=dict(arrowstyle="->"))

ax.annotate('Inner shape', family='helvetica',
            xy=(0.3, 0.25), xycoords='data', size=18,
            xytext=(-100, -90), textcoords='offset points',
            bbox=dict(boxstyle="round", fc='white', edgecolor='gray'),
            arrowprops=dict(arrowstyle="->"))

# ax.legend(ncol=2, loc='upper center')

ax.set_xticks(np.array([0,0.1,0.2,0.3,0.4,0.5]))
# ax.set_xlabel(r"$x \, (m)$")
# ax.set_ylabel(r"$y \, (m)$")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

plt.savefig('high_high_object.pdf', bbox_inches="tight", pad_inches=0.05, dpi=300)
plt.show()

# high high
# rec1 = pat.Rectangle(xy = (0.15, 0.15), width = 0.3, height = 0.3,
# angle = 0, color = "darkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.17, 0.17), width = 0.26, height = 0.26,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# high low
# rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# angle = 0, color = "darkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.22, 0.3), width = 0.16, height = 0.18,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# low high
# rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# angle = 0, color = "darkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.12, 0.12), width = 0.16, height = 0.26,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# low low
# rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# angle = 0, color = "daarkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.25, 0.25), width = 0.1, height = 0.1,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# decoboko
# rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# angle = 0, color = "darkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.15, 0.15), width = 0.1, height = 0.3,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# rec3 = pat.Rectangle(xy = (0.25, 0.15), width = 0.1, height = 0.15,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# rec4 = pat.Rectangle(xy = (0.35, 0.15), width = 0.1, height = 0.3,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')



#################################################################
# 目盛りなし
#################################################################

# import numpy as np
# import torch
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.patches as pat

# # import seaborn as sns
# # plt.style.use("ggplot")
# from matplotlib import cm
# import time


# def set_plt():
#     plt.xlim(0, 0.6)
#     plt.ylim(0, 0.6)
#     # plt.xlim(-0.2, 0.8)
#     # plt.ylim(-0.2, 0.8)
#     plt.xticks(color="None")
#     plt.yticks(color="None")
#     plt.tick_params(length=0)
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['bottom'].set_visible(False)
#     plt.axes().set_aspect('equal', 'datalim')

# fig = plt.figure(figsize=(3.0, 3.0), dpi=300)
# ax = fig.add_subplot(111)



# # high high
# rec1 = pat.Rectangle(xy = (0.15, 0.15), width = 0.3, height = 0.3,
# angle = 0, color = "darkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.17, 0.17), width = 0.26, height = 0.26,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# # high low
# # rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# # angle = 0, color = "darkred", alpha = 1, label='External shape')

# # rec2 = pat.Rectangle(xy = (0.22, 0.3), width = 0.16, height = 0.18,
# # angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# # low low
# rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# angle = 0, color = "darkred", alpha = 1, label='External shape')

# rec2 = pat.Rectangle(xy = (0.25, 0.25), width = 0.1, height = 0.1,
# angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# # decoboko
# # rec1 = pat.Rectangle(xy = (0.1, 0.1), width = 0.4, height = 0.4,
# # angle = 0, color = "darkred", alpha = 1, label='External shape')

# # rec2 = pat.Rectangle(xy = (0.15, 0.15), width = 0.1, height = 0.3,
# # angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# # rec3 = pat.Rectangle(xy = (0.25, 0.15), width = 0.1, height = 0.15,
# # angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')

# # rec4 = pat.Rectangle(xy = (0.35, 0.15), width = 0.1, height = 0.3,
# # angle = 0, color = "lightskyblue", alpha = 1, label='Internal shape')


# set_plt()

# ax.add_patch(rec1)
# ax.add_patch(rec2)
# # ax.add_patch(rec3)
# # ax.add_patch(rec4)

# # ax.annotate('Outer shape', family='helvetica',
# #             xy=(0.3, 0.45), xycoords='data', size=18,
# #             xytext=(-10, 40), textcoords='offset points',
# #             bbox=dict(boxstyle="round", fc='white', edgecolor='gray'),
# #             arrowprops=dict(arrowstyle="->"))

# # ax.annotate('Inner shape', family='helvetica',
# #             xy=(0.3, 0.25), xycoords='data', size=18,
# #             xytext=(-100, -90), textcoords='offset points',
# #             bbox=dict(boxstyle="round", fc='white', edgecolor='gray'),
# #             arrowprops=dict(arrowstyle="->"))

# # ax.legend(ncol=2, loc='upper center')

# # ax.set_xticks(np.array([0,0.1,0.2,0.3,0.4,0.5]))
# # ax.set_xlabel(r"$x \, (m)$")
# # ax.set_ylabel(r"$y \, (m)$")
# # ax.set_xlabel("x (m)")
# # ax.set_ylabel("y (m)")



# plt.savefig('low_low_object.pdf', pad_inches=0.05, dpi=300)
# plt.show()
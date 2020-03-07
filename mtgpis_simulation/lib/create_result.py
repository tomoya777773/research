#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 全体
plt.rcParams['figure.figsize']    = (7,5)
plt.rcParams['figure.dpi']        = 300
plt.rcParams['font.size']         = 24
# plt.rcParams['font.family']       = 'serif'
plt.rcParams['mathtext.fontset']  = 'stix' # math fontの設定
plt.rcParams['font.sans-serif'] = ['Helvetica']

# 軸
plt.rcParams['xtick.direction']   = 'in'
plt.rcParams['ytick.direction']   = 'in'
# plt.rcParams['xtick.labelsize']   = 15 # 軸だけ変更されます。
# plt.rcParams['ytick.labelsize']   = 15 # 軸だけ変更されます
plt.rcParams['xtick.major.width'] = 1.2 # x目盛の太さ
plt.rcParams['ytick.major.width'] = 1.2 # y目盛の太さ
plt.rcParams['xtick.major.size']  = 5 # x軸目盛りの長さ
plt.rcParams['ytick.major.size']  = 5 # y軸目盛りの長さ
plt.rcParams['axes.linewidth']    = 1.5 #軸の太さを設定。目盛りは変わらない
plt.rcParams['axes.grid']         = True
plt.rcParams['grid.linestyle']    = '--'
plt.rcParams['grid.linewidth']    = 0.3
# plt.figure(figsize=(4, 4), dpi=300)

# 凡例
# plt.rcParams["legend.markerscale"]   = 1
plt.rcParams["legend.fancybox"]      = False # 丸角
plt.rcParams["legend.framealpha"]    = 1 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"]     = 'black' # edgeの色を変更
plt.rcParams["legend.handlelength"]  = 1.0 # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"]  = 0.4 # 垂直方向（縦）の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 0.5 # 凡例の線と文字の距離の長さ

fig = plt.figure(figsize=(7,5), dpi=300)
plt.subplots_adjust(wspace=0.1)


name = "low_high"
file = "surf"
mtgpis  = np.load("../data/{}/mtgpis/value/{}.npy".format(name, file))
gpis    = np.load("../data/{}/gpis/value/{}.npy".format(name, file))
print(len(mtgpis))
# x = np.arange(len(mtgpis)) * 0.005 # low low
x = np.arange(len(mtgpis)) * 0.01
print(len(gpis))

# # # surface
# plt.plot(x, 1-gpis, c="red", linewidth=4, label="Tactile")
# plt.plot(x, 1-mtgpis, c="limegreen", linewidth=4, label="Visual-tactile")
# plt.xlabel("Travel length [m]")
# # plt.ylabel(r"$\eta_{\rm {area}}$")
# plt.ylabel("Area error")
# plt.ylim(-0.05, 1.05)

# # plt.xticks(np.array([0,0.4,0.8,1.2]))
# # plt.xticks(np.array([0,0.2,0.4,0.6,0.8,1.0]))
# plt.yticks(np.array([0,0.2,0.4,0.6,0.8,1.0]))

# plt.legend(loc='uper right')
# plt.tight_layout()
# plt.savefig('../data/graph/{}_{}.pdf'.format(name,file), bbox_inches="tight", pad_inches=0.05, dpi=300)

# plt.show()

# var
# plt.plot(x, gpis, c="red", linewidth=4, label="Tactile")
# plt.plot(x, mtgpis, c="limegreen", linewidth=4, label="Visual-tactile")
# plt.xlabel("Travel length [m]")
# plt.ylabel("Uncertainty measure")
# plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))#y軸小数点以下3桁表示

# # plt.xticks(np.array([0,0.4,0.8,1.2]))
# # plt.xticks(np.array([0,0.2,0.4,0.6,0.8,1.0]))

# # plt.yticks(np.array([0,2,4,6,8]))

# plt.legend(loc='uper right')

# plt.tight_layout()
# plt.savefig('../data/graph/{}_{}.pdf'.format(name,file), bbox_inches="tight", pad_inches=0.05, dpi=300)

# plt.show()


# # simirality
hh_s = np.load("../data/high_high/mtgpis/value/simirality.npy")
hl_s = np.load("../data/high_low/mtgpis/value/simirality.npy")
lh_s = np.load("../data/low_high/mtgpis/value/simirality.npy")
ll_s = np.load("../data/low_low/mtgpis/value/simirality.npy")
de_s = np.load("../data/decoboco/mtgpis/value/simirality.npy")

hh_s_t12 = []
hl_s_t12 = []
lh_s_t12 = []
ll_s_t12 = []
de_s_t12 = []

for simirality in hh_s:
    hh_s_t12.append(simirality[1][0] / simirality[1][1])

for simirality in hl_s:
    hl_s_t12.append(simirality[1][0] / simirality[1][1])

for simirality in lh_s:
    lh_s_t12.append(simirality[1][0] / simirality[1][1])

for simirality in ll_s:
    ll_s_t12.append(simirality[1][0] / simirality[1][1])

for simirality in de_s:
    de_s_t12.append(simirality[1][0] / simirality[1][1])

# plt.plot(np.arange(len(hh_s)) * 0.01, hh_s_t12, c="red", linewidth=4, label="Object A")
# plt.plot(np.arange(len(hl_s)) * 0.01, hl_s_t12, c="blueviolet", linewidth=4, label="Object B")
# # plt.plot(np.arange(len(lh_s)) * 0.01, lh_s_t12, c="hotpink", linewidth=4, label="Object C")
# plt.plot(np.arange(len(ll_s)) * 0.005, ll_s_t12, c="dodgerblue", linewidth=4, label="Object C")
# plt.plot(np.arange(len(de_s)) * 0.01, de_s_t12, c="limegreen", linewidth=4, label="Object D")

# for presen
plt.plot(np.arange(len(hh_s)) * 0.01, hh_s_t12, c="red", linewidth=4, label="Object A")
plt.plot(np.arange(len(de_s)) * 0.01, de_s_t12, c="limegreen", linewidth=4, label="Object B")
# plt.plot(np.arange(len(lh_s)) * 0.01, lh_s_t12, c="hotpink", linewidth=4, label="Object C")
plt.plot(np.arange(len(hl_s)) * 0.01, hl_s_t12, c="blueviolet", linewidth=4, label="Object C")
plt.plot(np.arange(len(ll_s)) * 0.01, ll_s_t12, c="dodgerblue", linewidth=4, label="Object D")



# plt.plot(np.arange(len(hh_s)) * 0.01, hh_s_t12, c="red", linewidth=4, label="Object1")
# plt.plot(np.arange(len(hl_s)) * 0.01, hl_s_t12, c="limegreen", linewidth=4, label="Object2")
# plt.plot(np.arange(len(lh_s)) * 0.01, lh_s_t12, c="hotpink", linewidth=4, label="Object3")
# plt.plot(np.arange(len(ll_s)) * 0.01, ll_s_t12, c="dodgerblue", linewidth=4, label="Object4")
# plt.plot(np.arange(len(de_s)) * 0.01, de_s_t12, c="blueviolet", linewidth=4, label="Object5")

plt.xlabel("Travel length [m]")
plt.ylabel("Similarity")
plt.legend(loc='lower right')
plt.xticks(np.array([0,0.5,1.0,1.5]))
plt.yticks(np.array([0,0.2,0.4,0.6,0.8,1.0]))

plt.tight_layout()
plt.savefig('../data/graph/simirality.pdf', bbox_inches="tight", pad_inches=0.05, dpi=300)

plt.show()
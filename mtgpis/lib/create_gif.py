#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
from PIL import Image
import os
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.animation as animation

model = "gpis"
object = "deco"

if __name__ == "__main__":

    folderName = "../data/{}/{}/movie/".format(object, model)

    #画像ファイルの一覧を取得
    picList = os.listdir(folderName)
    picList = natsorted(picList)
    print(picList)
    print(picList[0])
    #figオブジェクトを作る
    fig = plt.figure(figsize=(6.0, 6.0))

    #軸を消す
    ax = plt.subplot(1, 1, 1)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['bottom'].set_color('None')
    ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    #空のリストを作る
    ims = []

    #画像ファイルを順々に読み込んでいく
    for i in range(len(picList)):

        #1枚1枚のグラフを描き、appendしていく
        tmp = Image.open(folderName + picList[i])

        #エイリアシングを防ぐため、線形補完
        ims.append([plt.imshow(tmp)])

    #アニメーション作成
    ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=1000)

    ani.save('../movie/{}_{}.gif'.format(object, model), writer="imagemagick")
    # ani.save('anim.mp4', writer="ffmpeg")
    # plt.show()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def interpolate_data(data_2d):

    interpolate_num = 3
    interpolate_row = []
    interpolate_colum = []

    # interplate row
    for i in range(len(data_2d)):
        interpolate_row.append(data_2d[i])

        if i == len(data_2d) - 1 :
            break

        for j in range(1,interpolate_num):
            interpolate_row.append( data_2d[i] + (data_2d[i+1] - data_2d[i]) / interpolate_num * j )


    interpolate_row = np.array(interpolate_row)

    data_2d = np.reshape(interpolate_row, (-1, interpolate_row.shape[1])).T

    # interpolate colum
    for i in range(len(data_2d)):
        interpolate_colum.append(data_2d[i])

        if i == len(data_2d) - 1 :
            break

        for j in range(1, interpolate_num):
            interpolate_colum.append( data_2d[i] + (data_2d[i+1] - data_2d[i]) / interpolate_num * j )

    interpolate_colum = np.array(interpolate_colum).T


    return interpolate_colum.reshape((-1, interpolate_colum.shape[0]))


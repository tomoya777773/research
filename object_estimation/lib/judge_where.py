#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
現在位置が物体上か外か中か判定する
"""

def judge_ellipse(position, r):


    """
    Paraneters
    ----------
    position :  1d array
    r : radius

    Returns
    -------
    -1 : inside
    0  : on
    1  : outside

    """

    x,y,z = position
    judge = (x**2 / 4 + y**2 / 4 + z**2) - r**2
    if judge < 0: # in
        return -1
    elif judge > 0 and round(judge, 4) == 0: # on
        return 0
    else: # out
        return 1
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

object = "decoboco"

file_path = "../data/{}/shape_image".format(object)



inner = cv2.imread("{}/inner_shape.png".format(file_path))
outer = cv2.imread("{}/outer_shape.png".format(file_path))
inner = np.where(inner >= 126, 255, 0)
outer = np.where(outer >= 126, 255, 0)

h1, w1 = inner.shape[:2]
mask1  = np.zeros((h1 + 2, w1 + 2), dtype=np.uint8)
print(h1,w1)
_, inner, _, _ = cv2.floodFill(inner, mask1, seedPoint=(int(450), int(550)), newVal=(0, 0, 255))
cv2.imwrite("inner_tmp.png", inner)
inner = cv2.imread('inner_tmp.png')

h2, w2 = outer.shape[:2]
mask2  = np.zeros((h2 + 2, w2 + 2), dtype=np.uint8)
_, outer, _, _ = cv2.floodFill(outer, mask2, seedPoint=(int(450), int(450)), newVal=(0, 0, 255))
cv2.imwrite("outer_tmp.png", outer)
outer = cv2.imread('outer_tmp.png')

inner_cnt = 0
outer_cnt = 0
overlap_cnt = 0

for j in range(inner.shape[0]):
    for k in range(inner.shape[1]):
        if np.all(outer[j][k] == np.array([0,0,255])):
            outer_cnt += 1
            # if np.all(inner[j][k] == np.array([0,0,255])):
            #     overlap_cnt += 1
        if np.all(inner[j][k] == np.array([0,0,255])):
            inner_cnt += 1

print(inner_cnt)
print(outer_cnt)
print(overlap_cnt)
print("similarity:",inner_cnt/outer_cnt)


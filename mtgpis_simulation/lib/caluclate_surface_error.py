import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

model = "gpis"
object = "low_high"


picList = os.listdir("../data/{}/{}/estimate_surface/".format(object, model))


# image1
img1   = cv2.imread("../data/{}/{}/true_surface/step1.png".format(object,model))
img1   = np.where(img1 >= 126, 255, 0)
h1, w1 = img1.shape[:2]
mask1  = np.zeros((h1 + 2, w1 + 2), dtype=np.uint8)
print(w1)
_, img1, _, _ = cv2.floodFill(img1, mask1, seedPoint=(int(400), int(440)), newVal=(0, 0, 255))
cv2.imwrite("img1_tmp.png", img1)
img1 = cv2.imread('img1_tmp.png')

img1_cnt = 0
for j in range(img1.shape[0]):
    for k in range(img1.shape[1]):
        if np.all(img1[j][k] == np.array([0,0,255])):
            img1_cnt += 1

# image2
surf_list = []
for i in range(len(picList)):
    print("STEP:", i+1)
    img2   = cv2.imread("../data/{}/{}/estimate_surface/step{}.png".format(object, model, i+1))

    img2   = np.where(img2 >= 126, 255, 0)
    h2, w2 = img2.shape[:2]
    mask2  = np.zeros((h2 + 2, w2 + 2), dtype=np.uint8)
    print(w2)
    _, img2, _, _ = cv2.floodFill(img2, mask2, seedPoint=(int(320), int(320)), newVal=(0, 0, 255)) # w h
    # _, img2, _, _ = cv2.floodFill(img2, mask2, seedPoint=(int(350), int(350)), newVal=(0, 0, 255)) # low_high

    cv2.imwrite("img2_tmp.png", img2)
    img2 = cv2.imread('img2_tmp.png')

    img2_cnt = 0
    overlap_cnt = 0
    for j in range(img1.shape[0]):
        for k in range(img1.shape[1]):
            if np.all(img2[j][k] == np.array([0,0,255])):
                img2_cnt += 1
                if np.all(img1[j][k] == np.array([0,0,255])):
                    overlap_cnt += 1
            # if np.all( img1[j][k] + img2[j][k] == np.array([255,0,255]) ):
            #     overlap_cnt += 1

    print(img1_cnt)
    print(img2_cnt)
    print(overlap_cnt)

    if img1_cnt > img2_cnt:
        res = overlap_cnt / img1_cnt
    else:
        res = overlap_cnt / img2_cnt
    print(res)
    surf_list.append(res)

print(surf_list)
surf_list = np.array(surf_list)
np.save('../data/{}/{}/value/surf'.format(object, model), surf_list)
plt.plot(np.arange(len(surf_list)), surf_list)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
img=plt.imread('lenna.png')
img=img.mean(axis=-1)
#高斯平滑
blurred = cv2.GaussianBlur(img, (5, 5), 0.5)
#梯度
dx= cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
#非极大值抑制
img_tidu, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
img_yizhi = np.zeros(img_tidu.shape)

for i in range(1, img_tidu.shape[0]-1):
    for j in range(1, img_tidu.shape[1]-1):
        flag = True
        temp = img_tidu[i-1:i+2, j-1:j+2]
        if angle[i, j] <= -1:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
#双阈值检测
lower_boundary = img_tidu.mean() * 0.5
high_boundary = lower_boundary * 3
zhan = []
for i in range(1, img_yizhi.shape[0] - 1):
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:
            img_yizhi[i, j] = 0

while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255
        zhan.append([temp_1 - 1, temp_2 - 1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0
cv2.imshow('img', img_yizhi)
cv2.waitKey(0)



#透视变换
img1 = cv2.imread('photo1.jpg')
result1 = img1.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
m = cv2.getPerspectiveTransform(src, dst)
result = cv2.warpPerspective(result1, m, (337, 488))
cv2.imshow("src", img1)
cv2.imshow("result", result)
cv2.waitKey(0)
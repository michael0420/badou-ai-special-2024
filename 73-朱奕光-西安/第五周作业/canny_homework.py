import cv2
import numpy as np

'''
高斯平滑
'''
img = cv2.imread('lenna.png', 0)
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow('Blur', img_blur)

'''
Sobel算子
'''
img_sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
img_sobel_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
img_sobel = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)
cv2.imshow('sobel', img_sobel)

'''
非极大值抑制
'''
img_yizhi = np.zeros(img_sobel.shape)
dx, dy = img.shape
img_sobel = img_sobel.astype(float)
img_sobel_x = np.where(img_sobel_x == 0, 0.00000001, img_sobel_x)
angle = img_sobel_y / img_sobel_x

for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True
        temp = img_sobel[i - 1:i + 2, j - 1:j + 2]
        if angle[i, j] <= -1:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_sobel[i, j]
cv2.imshow('yizhi', img_yizhi)

'''
双阈值检测
'''
lower_boundary = img_sobel.mean() * 0.5
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
cv2.imshow('result', img_yizhi)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
Canny算法
'''
img_canny = cv2.Canny(img,100,300)
cv2.imshow('Canny', img_canny)
cv2.waitKey(0)
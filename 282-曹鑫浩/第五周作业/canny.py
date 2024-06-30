import numpy as np
import math
import cv2


# 高斯滤波
def gaussian_filter(img, dim, sigma):
    ftr = np.zeros((dim, dim))
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            ftr[i, j] = n1 * math.exp(n2 * (((i - dim)//2)**2 - ((j - dim)//2)**2))
    ftr /= (ftr.mean() * dim ** 2)
    img_pad = np.pad(img, ((dim//2, dim//2), (dim//2, dim//2)), 'constant')
    img_dst = np.zeros(img.shape)
    for i in range(img_dst.shape[0]):
        for j in range(img_dst.shape[1]):
            img_dst[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * ftr)
    return img_dst


# 求梯度方向
img = cv2.imread('lenna.png', 0)
img_src = gaussian_filter(img, 5, 0.5)
gradient_x = cv2.Sobel(img_src, cv2.CV_16S, 1, 0)
gradient_y = cv2.Sobel(img_src, cv2.CV_16S, 0, 1)
gradient_xy = cv2.addWeighted(cv2.convertScaleAbs(gradient_x), 0.5, cv2.convertScaleAbs(gradient_y), 0.5, 0)
gradient_x = gradient_x.astype(np.float64)
gradient_y = gradient_y.astype(np.float64)
for i in range(gradient_y.shape[0]):
    for j in range(gradient_y.shape[1]):
        if gradient_y[i, j] == 0:
            gradient_y[i, j] = 0.0000001
        else:
            continue
tan = gradient_x / gradient_y


# 非极大值抑制
img_yizhi = np.zeros(np.shape(img_src), gradient_xy.dtype)
for i in range(1, gradient_xy.shape[0] -1):
    for j in range(1, gradient_xy.shape[1] -1):
        temp = gradient_xy[i-1:i+2, j-1:j+2]
        temp = temp.astype(np.float64)
        if tan[i, j] >= 1:
            num1 = (temp[1, 0] - temp[0, 0]) / abs(tan[i, j]) + temp[1, 0]
            num2 = (temp[1, 2] - temp[2, 2]) / abs(tan[i, j]) + temp[1, 2]
            if temp[1, 1] >= num1 and temp[1, 1] >= num2:
                img_yizhi[i, j] = temp[1, 1]
            else:
                pass
        elif tan[i, j] <= -1:
            num1 = (temp[1, 0] - temp[2, 0]) / abs(tan[i, j]) + temp[1, 0]
            num2 = (temp[1, 2] - temp[0, 2]) / abs(tan[i, j]) + temp[1, 2]
            if temp[1, 1] >= num1 and temp[1, 1] >= num2:
                img_yizhi[i, j] = temp[1, 1]
            else:
                pass
        elif 0 < tan[i, j] < 1:
            num1 = (temp[0, 1] - temp[0, 0]) * abs(tan[i, j]) + temp[0, 1]
            num2 = (temp[2, 1] - temp[2, 2]) * abs(tan[i, j]) + temp[2, 1]
            if temp[1, 1] >= num1 and temp[1, 1] >= num2:
                img_yizhi[i, j] = temp[1, 1]
            else:
                pass
        elif -1 < tan[i, j] < 0:
            num1 = (temp[0, 1] - temp[0, 2]) * abs(tan[i, j]) + temp[0, 1]
            num2 = (temp[2, 1] - temp[2, 0]) * abs(tan[i, j]) + temp[2, 1]
            if temp[1, 1] >= num1 and temp[1, 1] >= num2:
                img_yizhi[i, j] = temp[1, 1]
            else:
                pass
print(img_yizhi)


# 双阈值检测
lower_boundary = gradient_xy.mean() * 0.5
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
    temp_1, temp_2 = zhan.pop()  # 出栈
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
        zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
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

cv2.imshow('boundary', img_yizhi)
cv2.waitKey(0)
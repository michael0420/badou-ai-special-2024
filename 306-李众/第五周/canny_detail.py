# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

img = plt.imread("lenna.png")
img = img * 255 
img = img.mean(axis=-1) #灰度化

#高斯滤波
sigma = 2
dim = 5
gauss = np.zeros([dim,dim])

tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列，这里直接拷贝参考代码的
n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核
n2 = -1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        gauss[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))#按给定的二维高斯分布生成5*5的高斯核
gauss = gauss/gauss.sum()#权重归一化操作，防止卷积之后图像整体亮度变化

dx, dy = img.shape
img_new = np.zeros(img.shape)  
tmp = dim//2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*gauss)
plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 类型转换
plt.axis('off')

#梯度
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
img_tidu_y = np.zeros([dx, dy])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
        img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
img_tidu_x[img_tidu_x == 0] = 0.00000001 #防止0值卡住
angle = img_tidu_y/img_tidu_x#梯度的角度方向计算
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

#非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx-1):
    for j in range(1, dy-1):
        flag = True  # 标记矩阵，为true则改为0
        temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
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
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

#双阈值算法检测、连接边缘
lower_boundary = img_tidu.mean() * 0.5
high_boundary = lower_boundary * 3  
zhan = []
for i in range(1, img_yizhi.shape[0]-1):  
    for j in range(1, img_yizhi.shape[1]-1):
        if img_yizhi[i, j] >= high_boundary:  
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:
            img_yizhi[i, j] = 0
 
while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop() 
    a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1-1, temp_2-1] = 255  
        zhan.append([temp_1-1, temp_2-1]) 
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
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  
plt.show()

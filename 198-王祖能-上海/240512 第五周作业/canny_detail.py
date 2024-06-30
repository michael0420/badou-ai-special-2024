'''
Canny边缘检测
@zeno wang
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import rgb2gray

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)  # 归一化的浮点型像素值读入
    if pic_path[-4:] == '.png':
        img = 255 * img
    # 1.图像灰度化
    img = rgb2gray(img)
    # img = img.mean(axis=-1)  # axis=-1就是最后一个通道取均值，不同灰度化方法，结果略有不同

    # 2.高斯滤波降噪
    sigma = 0.5  # 高斯核函数标准差，可调
    dim = 5  # 高斯核感受野选5x5是比较好的trade off
    radius = dim // 2  # 高斯核半径
    Gaussian_filter = np.zeros([dim, dim])
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = - 1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * ((i - radius) ** 2 + (j - radius) ** 2))
    Gaussian_filter /= Gaussian_filter.sum()
    print('变换后{} x {}的高斯卷积核为：\n{}'.format(dim, dim, Gaussian_filter))
    dx, dy = img.shape
    img_new = np.zeros(img.shape)
    img_pad = np.pad(img, ((radius, radius), (radius, radius)))
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    print('高斯滤波后的图像矩阵为：\n', img_new)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 255的浮点型数据，要强制类型转换
    plt.axis('off')

    # 3.sobel边缘提取。泰勒展开式，中心差分是二阶精度，前向和后向差分是一阶精度
    sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    img_grad_x = np.zeros(img_new.shape)
    img_grad_y = np.zeros(img_new.shape)
    img_grad = np.zeros(img_new.shape)
    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_x)
            img_grad_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_y)
            img_grad[i, j] = np.sqrt((img_grad_x[i, j] ** 2 + img_grad_y[i, j] ** 2))
    print('sobel后梯度矩阵为：\n', img_grad)
    img_grad[img_grad == 0] = 0.00000001
    angle = img_grad_y / img_grad_x
    plt.figure(2)
    plt.imshow(img_grad.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4.非极大值抑制
    img_suppression = np.zeros(img.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # True保留，False抹去
            if angle[i, j] >= 1:
                temp1 = img_grad[i-1, j] + (img_grad[i-1, j+1] - img_grad[i-1, j]) / angle[i, j]
                temp2 = img_grad[i+1, j] + (img_grad[i+1, j-1] - img_grad[i+1, j]) / angle[i, j]
                if not (img_grad[i, j] > temp1 and img_grad[i, j] > temp2):
                    flag = False
            elif angle[i, j] <= -1:
                temp1 = img_grad[i-1, j] + (img_grad[i-1, j-1] - img_grad[i-1, j]) / (- angle[i, j])
                temp2 = img_grad[i+1, j] + (img_grad[i+1, j+1] - img_grad[i+1, j]) / (- angle[i, j])
                if not (img_grad[i, j] > temp1 and img_grad[i, j] > temp2):
                    flag = False
            elif angle[i, j] >= 0:
                temp1 = img_grad[i, j+1] + (img_grad[i-1, j+1] - img_grad[i, j+1]) / 1 * angle[i, j]
                temp2 = img_grad[i, j-1] + (img_grad[i+1, j-1] - img_grad[i, j-1]) / 1 * angle[i, j]
                if not (img_grad[i, j] > temp1 and img_grad[i, j] > temp2):
                    flag = False
            elif angle[i, j] < 0:
                temp1 = img_grad[i, j-1] + (img_grad[i-1, j-1] - img_grad[i, j-1]) * (- angle[i, j])
                temp2 = img_grad[i, j+1] + (img_grad[i+1, j+1] - img_grad[i, j+1]) * (- angle[i, j])
                if not (img_grad[i, j] > temp1 and img_grad[i, j] > temp2):
                    flag = False
            if flag:
                img_suppression[i, j] = img_grad[i, j]
    print('非极大值抑制梯度复制后图像矩阵为：\n', img_suppression)
    plt.figure(3)
    plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 5.双阈值检测和连接边缘， 遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    low_boundary = 50
    high_boundary = 3 * low_boundary
    zhan = []  # 进栈，储存
    for i in range(1, dx-1):  # 不考虑外圈
        for j in range(1, dy-1):
            if img_suppression[i, j] < low_boundary:  # 非边缘舍弃
                img_suppression[i, j] = 0
            if img_suppression[i, j] > high_boundary:  # 强边缘保留
                img_suppression[i, j] = 255
                zhan.append([i, j])

    while not len(zhan) == 0:
        x, y = zhan.pop()  # 每个强边缘坐标出栈， 遍历周围8邻域，是否存在弱边缘
        if (img_suppression[x-1, y-1] < high_boundary) and (img_suppression[x-1, y-1] > low_boundary):
            img_suppression[x-1, y-1] = 255
            zhan.append([x-1, y-1])  # 与强边缘相邻的弱边缘保留为边缘，并且进栈
        if (img_suppression[x-1, y] < high_boundary) and (img_suppression[x-1, y] > low_boundary):
            img_suppression[x-1, y] = 255
            zhan.append([x-1, y])
        if (img_suppression[x-1, y+1] < high_boundary) and (img_suppression[x-1, y+1] > low_boundary):
            img_suppression[x-1, y+1] = 255
            zhan.append([x-1, y+1])
        if (img_suppression[x, y-1] < high_boundary) and (img_suppression[x, y-1] > low_boundary):
            img_suppression[x, y-1] = 255
            zhan.append([x, y-1])
        if (img_suppression[x, y+1] < high_boundary) and (img_suppression[x, y+1] > low_boundary):
            img_suppression[x, y+1] = 255
            zhan.append([x, y+1])
        if (img_suppression[x+1, y-1] < high_boundary) and (img_suppression[x+1, y-1] > low_boundary):
            img_suppression[x+1, y-1] = 255
            zhan.append([x+1, y-1])
        if (img_suppression[x+1, y] < high_boundary) and (img_suppression[x+1, y] > low_boundary):
            img_suppression[x+1, y] = 255
            zhan.append([x+1, y])
        if (img_suppression[x+1, y+1] < high_boundary) and (img_suppression[x+1, y+1] > low_boundary):
            img_suppression[x+1, y+1] = 255
            zhan.append([x+1, y+1])

    for i in range(img_suppression.shape[0]):
        for j in range(img_suppression.shape[1]):
            if img_suppression[i, j] != 0 and img_suppression[i, j] != 255:
                img_suppression[i, j] = 0

    print('进行双阈值限制和连接边缘检测后的图像矩阵为：\n', img_suppression)
    plt.figure(4)
    plt.imshow(img_suppression, cmap='gray')
    plt.axis('off')
    plt.show()

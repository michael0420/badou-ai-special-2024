# _*_ coding: UTF-8 _*_
# @Time: 2024/4/23 10:26
# @Author: iris
# @Email: liuhw0225@126.com
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def gray(img_path):
    """
    灰度化
    Gray(i,j) = 0.299 * R(i,j) + 0.587 * G(i,j) + 0.114 * B(i,j)
    :param img_path: 文件路径
    :return:
    """
    image = cv2.imread(img_path)

    W, H = image.shape[:2]
    img_gray = np.zeros([W, H], image.dtype)
    for i in range(W):
        for j in range(H):
            m = image[i, j]
            img_gray[i, j] = int(m[0] * 0.114 + m[1] * 0.587 + m[2] * 0.299)

    return img_gray


def smooth(image, sigma=1.4, kernel_size=5):
    """
    去除噪音 - 使用 5x5 的高斯滤波器
    H[i, j] = (1/(2*pi*sigma**2))*exp(-1/2*sigma**2((i-k-1)**2 + (j-k-1)**2))
    :param image: 灰度图
    :param sigma: sigma 可调
    :param kernel_size: 卷积核大小
    :return:
    """
    # 设置卷积核大小
    gaussian_filter = np.zeros([kernel_size, kernel_size])
    v1 = 1 / (2 * math.pi * sigma ** 2)
    v2 = -1 / (2 * sigma ** 2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            gaussian_filter[i, j] = math.exp(v2 * (np.square(i - 3) + np.square(j - 3))) / v1
    # 归一化处理
    gaussian_filter /= gaussian_filter.sum()

    tmp = kernel_size // 2
    # 为矩阵加上padding
    img_gaussian = np.pad(image, ((tmp, tmp), (tmp, tmp)), 'constant')
    # 定义卷积之后的图像信息
    img_new = np.zeros(image.shape)
    W, H = image.shape

    for i in range(W):
        for j in range(H):
            img_new[i, j] = np.sum(img_gaussian[i: i + kernel_size, j: j + kernel_size] * gaussian_filter)

    return img_new


def gradients(image):
    """
    计算梯度幅值（利用Soble边缘检测）
    :param image:
    :return:
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dx, dy = image.shape
    img_x = np.zeros(image.shape)
    img_y = np.zeros([dx, dy])
    img_sobel = np.zeros(image.shape)
    padding = np.pad(image, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_x[i, j] = np.sum(padding[i: i + 3, j:j + 3] * sobel_x)
            img_y[i, j] = np.sum(padding[i: i + 3, j:j + 3] * sobel_y)
            img_sobel[i, j] = np.sqrt(img_x[i, j] ** 2 + img_y[i, j] ** 2)
    img_x[img_x == 0] = 0.00000001
    tan = img_y / img_x
    # plt.figure(2)
    plt.imshow(img_sobel.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    return img_x, img_y, img_sobel, tan


def NMS(img_x, img_y, img_sobel):
    """
    非极大值抑制
    :param img_x: x方向梯度
    :param img_y: y方向梯度
    :param img_sobel: 梯度幅值
    :return:
    """
    d = np.copy(img_sobel)
    W, H = img_sobel.shape
    nms = np.copy(d)
    nms[0, :] = nms[W - 1, :] = nms[:, 0] = nms[:, H - 1] = 0
    # 边沿不计算
    for i in range(1, W - 1):
        for j in range(1, H - 1):
            # 如果当前梯度为0，则表示该点不是边缘
            if img_sobel[i, j] == 0:
                nms[i, j] = 0
            else:
                # x 方向导数
                gradX = img_x[i, j]
                # y 方向导数
                gradY = img_y[i, j]
                temp = d[i, j]
                # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                if np.abs(gradY) > np.abs(gradX):
                    # 权重
                    w = np.abs(gradX) / np.abs(gradY)
                    g2 = d[i - 1, j]
                    g4 = d[i + 1, j]
                    # 如果x,y 方向导数符号一致
                    # 像素点位置关系
                    # g1  g2
                    #     c
                    #     g4  g3
                    if gradX * gradX > 0:
                        g1 = d[i - 1, j - 1]
                        g3 = d[i + 1, j + 1]
                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #     g2  g1
                    #     c
                    # g3  g4
                    else:
                        g1 = d[i - 1, j + 1]
                        g3 = d[i + 1, j - 1]
                # 如果x方向梯度值比较大，则趋于x分量
                else:
                    w = np.abs(gradY) / np.abs(gradX)
                    g2 = d[i, j - 1]
                    g4 = d[i, j + 1]
                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    #      g3
                    # g2 c g4
                    # g1
                    if gradX * gradY > 0:
                        g1 = d[i + 1, j - 1]
                        g3 = d[i - 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    # g1
                    # g2 c g4
                    #      g3
                    else:
                        g1 = d[i - 1, j - 1]
                        g3 = d[i + 1, j + 1]
                temp1 = w * g1 + (1 - w) * g2
                temp2 = w * g3 + (1 - w) * g4
                if temp >= temp1 and temp >= temp2:
                    nms[i, j] = temp
                else:
                    nms[i, j] = 0

    return nms


def double_threshold(nms):
    """
    双阈值检测
    :param nms:
    :return:
    """
    W, H = nms.shape
    DT = np.zeros([W, H])

    # 定义高低阈值
    TL = 0.1 * np.max(nms)
    TH = 0.3 * np.max(nms)

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            # 双阈值选取
            if (nms[i, j] < TL):
                DT[i, j] = 0
            elif (nms[i, j] > TH):
                DT[i, j] = 1
            # 连接
            elif (nms[i - 1, j - 1:j + 1] < TH).any() or (
                    nms[i + 1, j - 1:j + 1].any() or (nms[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    return DT


def Canny(img_path):
    # 对图像进行灰度化
    img_gray = gray(img_path)
    cv2.imshow('gray', img_gray)
    # 对图像进行高斯滤波
    img_new = smooth(img_gray, 1.4, 5)
    # 计算梯度幅值，Soble边缘检测
    img_x, img_y, img_sobel, tan = gradients(img_new)
    # 非极大值抑制
    nms = NMS(img_x, img_y, img_sobel)
    cv2.imshow('nms', nms)
    # 双阈值检测
    return double_threshold(nms)


if __name__ == '__main__':
    img_path = '../data/lenna.png'
    dst = Canny(img_path)
    cv2.imshow('dst', dst)
    plt.show()
    cv2.waitKey()

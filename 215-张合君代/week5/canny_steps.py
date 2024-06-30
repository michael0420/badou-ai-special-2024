# -*- coding: utf-8 -*-
"""
@author: zhjd

"""
import math

import matplotlib.pyplot as plt
import numpy as np

# Global variables
SOBEL_KERNEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def gaussian_smoothing(img, sigma=0.5, dim=5):
    gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * gaussian_filter)
    return img_new


def compute_gradients(img):
    dx, dy = img.shape
    gradient_x = np.zeros(img.shape)
    gradient_y = np.zeros([dx, dy])
    gradient = np.zeros(img.shape)
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            gradient_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * SOBEL_KERNEL_X)
            gradient_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * SOBEL_KERNEL_Y)
            gradient[i, j] = np.sqrt(gradient_x[i, j] ** 2 + gradient_y[i, j] ** 2)
    gradient_x[gradient_x == 0] = 0.00000001
    angle = gradient_y / gradient_x
    return gradient, angle


def non_max_suppression(gradient, angle):
    dx, dy = gradient.shape
    non_max_suppressed = np.zeros(gradient.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = gradient[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (gradient[i, j] > num_1 and gradient[i, j] > num_2):
                    flag = False
            # Similar conditions for other cases...
            if flag:
                non_max_suppressed[i, j] = gradient[i, j]
    return non_max_suppressed


def edge_detection(non_max_suppressed, lower_boundary=None, high_boundary=None):
    if lower_boundary is None:
        lower_boundary = non_max_suppressed.mean() * 0.5
    if high_boundary is None:
        high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, non_max_suppressed.shape[0] - 1):
        for j in range(1, non_max_suppressed.shape[1] - 1):
            if non_max_suppressed[i, j] >= high_boundary:
                non_max_suppressed[i, j] = 255
                zhan.append([i, j])
            elif non_max_suppressed[i, j] <= lower_boundary:
                non_max_suppressed[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = non_max_suppressed[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            non_max_suppressed[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            non_max_suppressed[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            non_max_suppressed[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            non_max_suppressed[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            non_max_suppressed[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            non_max_suppressed[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            non_max_suppressed[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            non_max_suppressed[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
    return non_max_suppressed


if __name__ == '__main__':
    pic_path = 'alex.jpg'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.jpg':
        img = img * 255
    img = img.mean(axis=-1)

    img_smoothed = gaussian_smoothing(img)

    gradient, angle = compute_gradients(img_smoothed)

    non_max_suppressed = non_max_suppression(gradient, angle)

    img_edges = edge_detection(non_max_suppressed)

    plt.figure(1)
    plt.imshow(img_smoothed.astype(np.uint8), cmap='gray')
    plt.axis('off')

    plt.figure(2)
    plt.imshow(gradient.astype(np.uint8), cmap='gray')
    plt.axis('off')

    plt.figure(3)
    plt.imshow(non_max_suppressed.astype(np.uint8), cmap='gray')
    plt.axis('off')

    plt.figure(4)
    plt.imshow(img_edges.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

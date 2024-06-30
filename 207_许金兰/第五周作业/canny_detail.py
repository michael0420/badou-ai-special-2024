"""
@author: 207-xujinlan
边缘提取，手写实现canny
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def canny_detail(img_path, sigma, dim):
    """
    手写实现canny
    :param img_path: 图片路径
    :param sigma: 高斯核标准差
    :param dim: 高斯核尺寸
    :return:
    """
    img = plt.imread(img_path)
    # png图片是0-1之间的数据，如果是png图片，需要转化，乘以255
    if img_path[-4:] == '.png':
        img = img * 255
    # 1.图片灰度化
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3
    # 2.高斯平滑
    print('gaussion filter')
    Gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    # 生成高斯核
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    # 对图片进行高斯平滑处理
    dx, dy = img_gray.shape
    img_new = np.zeros(img_gray.shape)
    tmp = dim // 2
    # 边缘填充
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)

    # 2.检测图像中的水平、垂直和对角边缘
    print('sobel')
    # 这里用到sobel矩阵
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_gradient_x = np.zeros(img_new.shape)
    img_gradient_y = np.zeros([dx, dy])
    img_gradient = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_gradient_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_gradient_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_gradient[i, j] = np.sqrt(img_gradient_x[i, j] ** 2 + img_gradient_y[i, j] ** 2)
    img_gradient_x[img_gradient_x == 0] = 0.00000001  # 把0替换成很小的数值，在不影响结果的前提下又能做除法
    angle = img_gradient_y / img_gradient_x

    # 3、非极大值抑制
    print('image suppression')
    img_suppression = np.zeros(img_gradient.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_gradient[i - 1:i + 2, j - 1:j + 2]
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_gradient[i, j] > num_1 and img_gradient[i, j] > num_2):
                    flag = False
            if flag:
                img_suppression[i, j] = img_gradient[i, j]

    # 4.双阈值检测，连接边缘。
    print('image double threshold')
    img_dthreshold = img_suppression.copy()
    lower_boundary = img_gradient.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img_dthreshold.shape[0] - 1):
        for j in range(1, img_dthreshold.shape[1] - 1):
            if img_dthreshold[i, j] >= high_boundary:
                img_dthreshold[i, j] = 255
                zhan.append([i, j])
            elif img_dthreshold[i, j] <= lower_boundary:
                img_dthreshold[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()
        a = img_dthreshold[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_dthreshold[temp_1 - 1, temp_2 - 1] = 255
            zhan.append([temp_1 - 1, temp_2 - 1])
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_dthreshold[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_dthreshold[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_dthreshold[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_dthreshold[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_dthreshold[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_dthreshold[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_dthreshold[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_dthreshold.shape[0]):
        for j in range(img_dthreshold.shape[1]):
            if img_dthreshold[i, j] != 0 and img_dthreshold[i, j] != 255:
                img_dthreshold[i, j] = 0
    return img_new, img_gradient, img_suppression, img_dthreshold


if __name__ == '__main__':
    img_path = 'lenna.png'
    sigma = 1.2
    dim = 5
    img_new, img_gradient, img_suppression, img_dthreshold = canny_detail(img_path, sigma, dim)
    plt.figure(1)
    plt.title('gaussion filter')
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(2)
    plt.title('image gradient')
    plt.imshow(img_gradient.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(3)
    plt.title('image suppression')
    plt.imshow(img_suppression.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(4)
    plt.title('image dthreshold')
    plt.imshow(img_dthreshold.astype(np.uint8), cmap='gray')
    plt.axis('off')

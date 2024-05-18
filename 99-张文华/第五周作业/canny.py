import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == '__main__':
    # 1、灰度化图像
    img_gray = cv2.imread('lenna.png', 0)
    plt.figure(1)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')

    # 2、给图像降噪，高斯平滑
    sigma = 1
    filter_dim = 5
    gaussian_filter = np.zeros([filter_dim, filter_dim])
    tmp = [i - filter_dim // 2 for i in range(filter_dim)]
    n1 = 1 / (2*math.pi*sigma**2)
    n2 = -1 / (2*sigma**2)
    for i in range(filter_dim):
        for j in range(filter_dim):
            gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    dx, dy = img_gray.shape
    img_gaussian = np.zeros(img_gray.shape)
    tmp = filter_dim // 2
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_gaussian[i, j] = np.sum(img_pad[i:i+filter_dim, j:j+filter_dim] * gaussian_filter)
    plt.figure(2)
    plt.imshow(img_gaussian, cmap='gray')
    plt.axis('off')

    # 3、对图像进行边缘检测，并求出每个像素点位置梯度,使用sobel算子
    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelY = np.array(([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    img_tiduX = np.zeros(img_gray.shape)
    img_tiduY = np.zeros(img_gray.shape)
    img_tidu = np.zeros(img_gray.shape)
    img_pad = np.pad(img_gray, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tiduX[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobelX)
            img_tiduY[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobelY)
            img_tidu[i, j] = np.sqrt(img_tiduX[i, j]**2 + img_tiduY[i, j]**2)
    img_tiduX[img_tiduX == 0] = 0.000001
    angle = img_tiduY / img_tiduX
    plt.figure(3)
    plt.imshow(img_tidu, cmap='gray')
    plt.axis('off')

    # 4、梯度非极大值抑制
    img_nms = np.zeros(img_gray.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
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
                img_nms[i, j] = img_tidu[i, j]
    plt.figure(4)
    plt.imshow(img_nms, cmap='gray')
    plt.axis('off')

    # 5、双阈值检测算法和边缘链接
    # 双阈值
    lower_boundary = img_tidu.mean()
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img_nms.shape[0] - 1):
        for j in range(1, img_nms.shape[1] - 1):
            if img_nms[i, j] >= high_boundary:
                img_nms[i, j] = 255
                zhan.append([i, j])
            elif img_nms[i, j] <= lower_boundary:
                img_nms[i, j] = 0

    # 边缘检测
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_nms[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_nms[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_nms[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_nms[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_nms[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_nms[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_nms[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_nms[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_nms[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_nms.shape[0]):
        for j in range(img_nms.shape[1]):
            if img_nms[i, j] != 0 and img_nms[i, j] != 255:
                img_nms[i, j] = 0

    plt.figure(5)
    plt.imshow(img_nms, cmap='gray')
    plt.axis('off')

    plt.show()
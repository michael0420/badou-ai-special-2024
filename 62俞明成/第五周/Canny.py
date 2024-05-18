import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

if __name__ == '__main__':
    # cv:BGR plt:RGB
    # img1 = cv.imread('../lenna.png')
    img2 = plt.imread('../lenna.png')
    img2 = img2 * 255
    img2 = img2.mean(axis=-1)

    # 高斯平滑
    dim = 5
    sigma = 0.5
    Gaussian_filter = np.zeros([dim, dim])
    # 中心偏移列表temp，用来计算temp[i] ** 2 + temp[j] ** 2部分
    temp = [i - dim // 2 for i in range(dim)]
    # print(temp)
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (temp[i] ** 2 + temp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img2.shape
    # 存储高斯平滑后的img
    img_new = np.zeros(img2.shape)
    zoom = dim // 2
    # np.pad(需要填充的数组,((上,下),(左,右)))
    img_pad = np.pad(img2, ((zoom, zoom), (zoom, zoom)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 梯度
    Sobel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobel_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    img_x = np.zeros(img_new.shape)
    img_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_tidu_pad = np.pad(img_new, ((1, 1), (1, 1)), mode='constant')
    for i in range(dx):
        for j in range(dy):
            img_x[i, j] = np.sum(img_tidu_pad[i:i + 3, j:j + 3] * Sobel_X)
            img_y[i, j] = np.sum(img_tidu_pad[i:i + 3, j:j + 3] * Sobel_Y)
            img_tidu[i, j] = np.sqrt(img_x[i, j] ** 2 + img_y[i, j] ** 2)
    img_x[img_x == 0] = 0.00000001
    angle = img_y / img_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    # 不算边缘
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # True保留，False不保留
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            # 使用线性插值法判断抑制与否
            if angle[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif angle[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
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
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    # 读取图像
    img_path = "lenna.png"
    img = plt.imread(img_path)
    if img_path[-4:] == '.png':
        img = img * 255
    # 1、对图像进行灰度化
    img = img.mean(axis=-1)

    # 2、高斯平滑
    sigma = 0.5
    # mean=0.5
    dim = 5
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    Gaussian_Kernel = np.zeros((dim, dim))
    tmp = np.zeros(dim)
    for i in range(dim):
        tmp[i] = i - dim // 2
    for i in range(dim):
        for j in range(dim):
            Gaussian_Kernel[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 归一化
    Gaussian_Kernel = Gaussian_Kernel / Gaussian_Kernel.sum()
    # print(Gaussian_Kernel)
    dx, dy = img.shape[:2]
    # 边缘填充
    img_pad = np.pad(img, pad_width=dim // 2, mode='constant', constant_values=0)
    # print(img)
    # print(img_new)

    # 对img图像进行高斯平滑滤波,img_new存储平滑之后的图像
    img_new = np.zeros(img.shape)
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_Kernel)
    # print(img)
    # print(img_new)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    # 3、检测提取图像中的水平，垂直，对角边缘（sobel算法，prewitt算法）
    # sobel卷积核
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 存放对应像素经过sobel卷积之后的梯度的矩阵
    img_tidu_x = np.zeros(img_new.shape)
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    # 边缘填充
    # img_pad = np.pad(img_new, ((1, 1), (1, 1)), mode='constant')
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    # 通过sobel算法求出每一个像素卷积之后的梯度
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.0000000001
    tan = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # cv2.imshow("img_tidu",img_tidu)
    # cv2.waitKey(0)

    # 4、对梯度幅值进行非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    # img_pad = np.pad(img_tidu, ((1, 1), (1, 1)), mode='constant')
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            # 取出八邻域
            # tmp = img_pad[i:i + 3, j:j + 3]
            tmp = img_tidu[i-1:i+2, j-1:j + 2]
            # 画图，利用双线性插值法求各位置的像素值，再确认该点是否是极大值点
            if tan[i, j] >= 1:
                num1 = (tmp[0, 2] - tmp[0, 1]) / tan[i, j] + tmp[0, 1]
                num2 = (tmp[2, 0] - tmp[2, 1]) / tan[i, j] + tmp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif tan[i, j] >= 0:
                num1 = (tmp[0, 2] - tmp[1, 2]) * tan[i, j] + tmp[1, 2]
                num2 = (tmp[2, 0] - tmp[1, 0]) * tan[i, j] + tmp[1, 0]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif tan[i, j] <= -1:
                num1 = (tmp[0, 1] - tmp[0, 0]) / tan[i, j] + tmp[0, 1]
                num2 = (tmp[2, 1] - tmp[2, 2]) / tan[i, j] + tmp[2, 1]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            elif tan[i, j] < 0:
                num1 = (tmp[1, 0] - tmp[0, 0]) * tan[i, j] + tmp[1, 0]
                num2 = (tmp[1, 2] - tmp[2, 2]) * tan[i, j] + tmp[1, 2]
                if not (img_tidu[i, j] > num1 and img_tidu[i, j] > num2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    # 5、用双阈值算法检测
    lower_threshold = img_tidu.mean() * 0.5
    high_threshold = lower_threshold*3
    zhan = []
    # 先找出强边缘点，然后根据强边缘点的八邻域判断是否有弱边缘点
    # 如果有，那么对于该弱边缘点来说，它的八邻域一定存在强边缘点，就是我，
    # 那么这个弱边缘点就可以保留，并且将像素值改成255，那么图像中剩余的
    # 没有跟强边缘点相邻的，而且像素值又比较大的，那就是孤立噪声点，到最后
    # 可以把它去掉
    for i in range(1, img_yizhi.shape[0]-1):
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_threshold:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_threshold:
                img_yizhi[i, j] = 0
    # 6、抑制孤立低阈值点
    # img_pad = np.pad(img_yizhi, ((1, 1), (1, 1)), mode='constant')
    while not len(zhan) == 0:
        x, y = zhan.pop()
        # 这里曾经写成img_pad[x - 1:x + 2, y - 1:y + 2],修改时没有改干净
        # tmp = img_yizhi[x - 1:x + 2, y - 1:y + 2]
        tmp = img_yizhi[x - 1:x + 2, y - 1:y + 2]
        if lower_threshold < tmp[0, 0] < high_threshold:
            img_yizhi[x - 1, y - 1] = 255
            zhan.append([x - 1, y - 1])
        if lower_threshold < tmp[0, 1] < high_threshold:
            img_yizhi[x - 1, y] = 255
            zhan.append([x - 1, y])
        if lower_threshold < tmp[0, 2] < high_threshold:
            img_yizhi[x - 1, y + 1] = 255
            zhan.append([x - 1, y + 1])
        if lower_threshold < tmp[1, 0] < high_threshold:
            img_yizhi[x, y - 1] = 255
            zhan.append([x, y - 1])
        if lower_threshold < tmp[1, 2] < high_threshold:
            img_yizhi[x, y + 1] = 255
            zhan.append([x, y + 1])
        if lower_threshold < tmp[2, 0] < high_threshold:
            img_yizhi[x + 1, y - 1] = 255
            zhan.append([x + 1, y - 1])
        if lower_threshold < tmp[2, 1] < high_threshold:
            img_yizhi[x + 1, y] = 255
            zhan.append([x + 1, y])
        if lower_threshold < tmp[2, 2] < high_threshold:
            img_yizhi[x + 1, y + 1] = 255
            zhan.append([x + 1, y + 1])
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                # 那么它是噪声点，可以把它去掉
                img_yizhi[i, j] = 0
    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()

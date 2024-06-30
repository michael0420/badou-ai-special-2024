import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray(img_path):
    """
     img_gray[i,j] = int(B*0.11 + G*0.59 + R*0.3)
    :param img_path: 文件路径
    :return:
    """
    img = cv2.imread(img_path)
    if img_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print('------------------转换完成------------------')
    return img_gray


def Gaussian_smooth(img, sigma=1, kernel_size=5):
    """
    高斯平滑
    :param img: 灰度图
    :param sigma: 高斯平滑时的高斯核参数，标准差，可调
    :param kernel_size: 卷积核大小
    :return:
    """
    # 生成一个5x5的零矩阵，存储高斯核
    Gaussian_filter = np.zeros([kernel_size, kernel_size])
    # 生成一个序列
    tmp = [i-kernel_size//2 for i in range(kernel_size)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(kernel_size):
        for j in range(kernel_size):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 归一化处理
    Gaussian_filter /= Gaussian_filter.sum()
    dx, dy = img.shape
    # 存储平滑之后的图像
    img_new = np.zeros(img.shape)
    # 为矩阵加上padding 边缘填补
    tmp = kernel_size // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
    # 卷积之后的图像信息
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+kernel_size, j:j+kernel_size] * Gaussian_filter)
    print('------------------高斯平滑完成------------------')
    return img_new


def gradients(img):
    """
    求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    :param img:
    :return:
    """
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 存储梯度图像
    dx, dy = img.shape
    img_tidu_x = np.zeros(img.shape)
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img.shape)
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    tan = img_tidu_y / img_tidu_x
    print('------------------梯度完成------------------')
    return img_tidu, tan


def NMS(img_tidu, tan):
    """
    非极大值抑制
    :param img_x: x方向梯度
    :param img_y: y方向梯度
    :param img_sobel: 梯度幅值
    :return:
    """
    img_yizhi = np.zeros(img_tidu.shape)
    dx, dy = img_tidu.shape
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
            if tan[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / tan[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / tan[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * tan[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * tan[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    print('------------------非极大值抑制完成------------------')
    return img_yizhi


def double_threshold(img_yizhi, img_tidu):
    """
    双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    :param img_yizhi:
    :param img_tidu:
    :return:
    """
    lower_boundary = img_tidu.mean() * 0.5
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
    print('------------------双阈值检测完成------------------')
    return img_yizhi


def Canny(img_path):
    # 对图像进行灰度化
    img_gray = gray(img_path)
    cv2.imshow('gray', img_gray)
    # 对图像进行高斯平滑
    img = Gaussian_smooth(img_gray, 1, 5)
    # 计算梯度幅值，Soble边缘检测
    img_tidu, tan = gradients(img)
    # 非极大值抑制
    img_yizhi = NMS(img_tidu, tan)
    cv2.imshow('nms', img_yizhi)
    # 双阈值检测
    return double_threshold(img_yizhi, img_tidu)


if __name__ == '__main__':
    img_path = '../../imgs/lenna.png'
    dst = Canny(img_path)
    print("--------------")
    cv2.imshow('dst', dst)
    plt.show()
    cv2.waitKey()
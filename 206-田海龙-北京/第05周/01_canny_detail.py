import math
import matplotlib.pyplot as plt
import numpy as np

import cv2

from utils import current_directory


def to_gray(img, img_path):
    """
    将图片转换为灰度图
    :param img: 输入的图片
    :param img_path: 图片的路径
    :return: 灰度图
    """

    # print("image", img)
    if (
        img_path[-4:] == ".png"
    ):  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img_gray = img.mean(axis=-1)  # 取均值的方法进行灰度化

    return img_gray


def to_gause(img, sigma=0.5, dim=5):
    """
    高斯滤波
    :param img: 输入的图片
    :param sigma: 高斯核的标准差
    :param dim: 高斯核的尺寸
    :return: 高斯滤波后的图片
    """

    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5  # 高斯核尺寸
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    # print(tmp)
    n1 = 1 / (2 * math.pi * sigma**2)  # 计算高斯核
    n2 = -1 / (2 * sigma**2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))

    # print(Gaussian_filter)
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_gause = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), "constant")  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_gause[i, j] = np.sum(
                img_pad[i : i + dim, j : j + dim] * Gaussian_filter
            )

    return img_gause


def to_guase_02(img, sigma=0.5, dim=5):
    """
    高斯滤波，cv2封装实现
    :param img: 输入的图片
    :param sigma: 高斯核的标准差
    :param dim: 高斯核的尺寸
    :return: 高斯滤波后的图片
    """
    img_gause = cv2.GaussianBlur(img, (dim, dim), sigma)

    return img_gause


def to_sobel(img):
    """
    sobel边缘提取
    :param img: 输入的图片
    :return: sobel边缘提取后的图片
    """

    dx, dy = img.shape
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobel_x = np.zeros(img.shape)  # 存储梯度图像
    img_sobel_y = np.zeros([dx, dy])
    img_sobel = np.zeros(img.shape)
    img_pad = np.pad(
        img, ((1, 1), (1, 1)), "constant"
    )  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_sobel_x[i, j] = np.sum(
                img_pad[i : i + 3, j : j + 3] * sobel_kernel_x
            )  # x方向
            img_sobel_y[i, j] = np.sum(
                img_pad[i : i + 3, j : j + 3] * sobel_kernel_y
            )  # y方向
            img_sobel[i, j] = np.sqrt(img_sobel_x[i, j] ** 2 + img_sobel_y[i, j] ** 2)

    img_sobel_x[img_sobel_x == 0] = 0.00000001
    # 正切
    tan = img_sobel_y / img_sobel_x

    return img_sobel, tan


def nms(img, tan):
    """
    非极大值抑制
    :param img: 输入的图片
    :param tan: 梯度方向
    :param out_shape: 输出的图片尺寸
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :return: 非极大值抑制后的图片
    """

    dx, dy = img.shape
    img_yizhi = np.zeros(img.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img[i - 1 : i + 2, j - 1 : j + 2]  # 梯度幅值的8邻域矩阵

            # 补充解释：
            # 一定要注意图像矩阵格式h，w，chanel，即高、宽、通道
            # 这样某个像素点的表示就很明确 img[i,j] i是行（高），j是列（宽）
            # 基于以上基础，下面代码中各个if判断中的坐标点就比较明确了，找的都是切线与坐标轴的交点

            # 再一个：对于正切值为负的情况，减法是反过来计算，即近值点-远值点（对角点）
            # 后面加的值，则是+近值点，从而计算沿着梯度方向的值

            if tan[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / tan[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan[i, j] + temp[2, 1]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif tan[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / tan[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan[i, j] + temp[2, 1]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif tan[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * tan[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan[i, j] + temp[1, 0]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            elif tan[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * tan[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan[i, j] + temp[1, 2]
                if not (img[i, j] > num_1 and img[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img[i, j]

    return img_yizhi


def twe_threshold(img_yizhi, low_threshold, high_threshold):
    """
    双阈值检测和连接边缘
    """

    # lower_boundary = img_tidu.mean() * 0.5
    # high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_threshold:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= low_threshold:  # 舍
                img_yizhi[i, j] = 0

    # while not len(zhan) == 0:
    #     temp_1, temp_2 = zhan.pop()  # 出栈
    #     a = img_yizhi[temp_1 - 1 : temp_1 + 2, temp_2 - 1 : temp_2 + 2]
    #     if (a[0, 0] < high_threshold) and (a[0, 0] > low_threshold):
    #         img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
    #         zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
    #     if (a[0, 1] < high_threshold) and (a[0, 1] > low_threshold):
    #         img_yizhi[temp_1 - 1, temp_2] = 255
    #         zhan.append([temp_1 - 1, temp_2])
    #     if (a[0, 2] < high_threshold) and (a[0, 2] > low_threshold):
    #         img_yizhi[temp_1 - 1, temp_2 + 1] = 255
    #         zhan.append([temp_1 - 1, temp_2 + 1])
    #     if (a[1, 0] < high_threshold) and (a[1, 0] > low_threshold):
    #         img_yizhi[temp_1, temp_2 - 1] = 255
    #         zhan.append([temp_1, temp_2 - 1])
    #     if (a[1, 2] < high_threshold) and (a[1, 2] > low_threshold):
    #         img_yizhi[temp_1, temp_2 + 1] = 255
    #         zhan.append([temp_1, temp_2 + 1])
    #     if (a[2, 0] < high_threshold) and (a[2, 0] > low_threshold):
    #         img_yizhi[temp_1 + 1, temp_2 - 1] = 255
    #         zhan.append([temp_1 + 1, temp_2 - 1])
    #     if (a[2, 1] < high_threshold) and (a[2, 1] > low_threshold):
    #         img_yizhi[temp_1 + 1, temp_2] = 255
    #         zhan.append([temp_1 + 1, temp_2])
    #     if (a[2, 2] < high_threshold) and (a[2, 2] > low_threshold):
    #         img_yizhi[temp_1 + 1, temp_2 + 1] = 255
    #         zhan.append([temp_1 + 1, temp_2 + 1])

    # 循环当前点的周围点，优化为循环操作
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1 : temp_1 + 2, temp_2 - 1 : temp_2 + 2]

        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue

                if (a[i, j] < high_threshold) and (a[i, j] > low_threshold):
                    # 这个像素点标记为边缘
                    img_yizhi[temp_1 - 1 + i, temp_2 - 1 + j] = 255
                    zhan.append([temp_1 - 1+ i, temp_2 - 1+ j])  # 进栈

    # 最后把不是0不是255的像素置为0
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    return img_yizhi


img_path = current_directory + "\\img\\lenna.png"


def canny_detail(img,low_threshold = None, high_threshold = None):
    """
    canny边缘检测
    :param img: 输入的图片
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :return: 边缘检测后的图片
    """

    # 过程：
    # 1）灰度化
    # 2）高斯滤波
    # 3）sobel边缘提取
    # 4）非极大值抑制
    # 5）双阈值检测和连接边缘

    img_gray = to_gray(img, img_path)

    plt.figure(1)
    plt.imshow(img_gray, cmap="gray")
    # plt.show()

    sigma = 0.5
    ksize = 5
    img_gause = to_gause(img_gray, sigma, ksize)
    img_gause = to_guase_02(img_gray, sigma, ksize)

    plt.figure(2)
    plt.imshow(img_gause, cmap="gray")
    # plt.show()

    img_sobel, tan = to_sobel(img_gause)

    plt.figure(3)
    plt.imshow(img_sobel, cmap="gray")
    # plt.show()

    img_yizhi = nms(img_sobel, tan)

    plt.figure(4)
    plt.imshow(img_yizhi, cmap="gray")
    # plt.show()

    if(low_threshold == None or high_threshold == None):
        low_threshold = img_sobel.mean() * 0.5
        high_threshold = low_threshold * 3

    img_res = twe_threshold(img_yizhi, low_threshold, high_threshold)

    plt.figure(5)
    plt.imshow(img_res.astype(np.uint8), cmap="gray")
    plt.show()

def canny_test():
    img = plt.imread(img_path)
    canny_detail(img,100,200)

canny_test()


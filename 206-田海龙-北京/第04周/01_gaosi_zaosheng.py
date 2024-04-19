import numpy as np
import cv2
import random


def gaosi_zaosheng(img, means, sigma, percetage):
    """
    高斯噪声
    :param img: 输入图像
    :param means: 高斯噪声均值
    :param sigma: 高斯噪声方差
    :param percetage: 高斯噪声百分比
    :return:
    """
    img_gaosi = img.copy()
    h, w = img.shape[0:2]

    # 需要增加高斯噪声的像素点数
    zaosheng_num = int(h * w * percetage)

    # 针对每个需要增加高斯噪声的像素点数，随机坐标点幅值高斯噪声
    for i in range(zaosheng_num):
        randow_x = random.randint(0, h - 1)
        random_y = random.randint(0, w - 1)
        gaosi_val = img_gaosi[randow_x, random_y] + random.gauss(means, sigma)
        if gaosi_val > 255:
            gaosi_val = 255
        elif gaosi_val < 0:
            gaosi_val = 0

        img_gaosi[randow_x, random_y] = gaosi_val

    return img_gaosi


from utils import cv_imread
import matplotlib.pyplot as plt

img_path = r"./206-田海龙-北京/第04周/img/lenna.png"


def test():
    img = cv_imread(img_path, 0)

    img_gaosi = gaosi_zaosheng(img, 2, 4, 0.8)

    # cv2.imshow("img",img)
    cv2.imshow("img_gaosi", img_gaosi)

    img = cv_imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("img", img)

    cv2.waitKey()

    # plt.subplot(121)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(img_gaosi, cmap='gray')

    # plt.savefig("./206-田海龙-北京/第04周/gaosi_zaosheng.png")

    # plt.show()


test()

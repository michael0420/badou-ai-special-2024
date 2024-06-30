import cv2
import random
import matplotlib.pyplot as plt

from utils import cv_imread

img_path = "./206-田海龙-北京/第04周/img/lenna.png"


def jiaoyan_zaosheng(img, percentage):
    """
    椒盐噪声
    :param img:原图
    :param percentage:噪声比例
    """

    h, w = img.shape[:2]

    zaosheng_num = int(h * w * percentage)

    for i in range(zaosheng_num):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        r = random.random()
        if r > 0.5:
            img[x, y] = 255
        else:
            img[x, y] = 0

    return img


def jiaoyan_zaosheng_rgb(img, percentage):
    """
    椒盐噪声-RGB三通道
    :param img:原图
    :param percentage:噪声比例
    """

    h, w, c = img.shape

    zaosheng_num = int(h * w * percentage)

    for i in range(zaosheng_num):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)

        for j in range(c):
            # 每个通道随机椒盐，更加合理
            r = random.random()
            if r > 0.5:
                img[x, y, j] = 255
            else:
                img[x, y, j] = 0

    return img


def jiaoyan_test():
    img = cv_imread(img_path)

    # xx=img[1,2]
    # print(xx)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_jiaoyan = jiaoyan_zaosheng(img, 0.05)

    plt.imshow(img_jiaoyan, cmap="gray")
    plt.show()


def jiaoyan_rgb_test():
    img = cv_imread(img_path)

    # xx=img[1,2]
    # print(xx)

    img_jiaoyan = jiaoyan_zaosheng_rgb(img, 0.1)
    img_jiaoyan = cv2.cvtColor(img_jiaoyan, cv2.COLOR_BGR2RGB)
    plt.imshow(img_jiaoyan)
    plt.show()


jiaoyan_test()
jiaoyan_rgb_test()

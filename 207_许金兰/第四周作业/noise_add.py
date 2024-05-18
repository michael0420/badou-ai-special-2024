"""
@author: 207-xujinlan
噪声添加，给图片加上高斯噪声或椒盐噪声
"""

import cv2
import random


class Noise_Add:
    """ 噪声添加 """

    def __init__(self, img):
        """
        初始化
        :param img: 输入图片，单通道或三通道
        """
        self.img_gauss = img.copy()  # 高斯噪声
        self.img_pepsalt = img.copy()  # 椒盐噪声
        self.w, self.h = img.shape[0], img.shape[1]
        if len(img.shape) == 2:
            self.c = None
        elif len(img.shape) == 3 and img.shape[2] == 3:
            self.c = img.shape[2]
        else:
            print('输入图片不是单通道或三通道，请重新输入图片。')

    def gauss_noise(self, mu, sigma, p):
        """
        高斯噪声
        :param mu: random.gauss均值
        :param sigma: random.gauss标准差
        :param p: 噪声比例
        :return:
        """
        for i in range(int(self.w * self.h * p)):
            # 随机选择要添加噪声的点
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if self.c is None:
                self.img_gauss[x, y] = self.img_gauss[x, y] + random.gauss(mu, sigma)    #添加高斯噪声
                # 截取小于0和大于255的点
                if self.img_gauss[x, y] < 0:
                    self.img_gauss[x, y] = 0
                elif self.img_gauss[x, y] > 255:
                    self.img_gauss[x, y] = 255
            elif self.c == 3:
                z = random.randint(0, self.c - 1)
                self.img_gauss[x, y, z] = self.img_gauss[x, y, z] + random.gauss(mu, sigma)
                if self.img_gauss[x, y, z] < 0:
                    self.img_gauss[x, y, z] = 0
                elif self.img_gauss[x, y, z] > 255:
                    self.img_gauss[x, y, z] = 255
            else:
                print('输入图片不是单通道或三通道，请重新输入图片。')

    def pepper_salt_noise(self, p):
        """
        椒盐噪声
        :param p: 噪声比例
        :return:
        """
        for i in range(int(self.w * self.h * p)):
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if self.c is None:
                self.img_pepsalt[x, y] = random.choice([0, 255])
            elif self.c == 3:
                z = random.randint(0, self.c - 1)
                self.img_pepsalt[x, y, z] = random.choice([0, 255])
            else:
                print('输入图片不是单通道或三通道，请重新输入图片。')


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 1)
    add_noise = Noise_Add(img)  # 实例化
    add_noise.gauss_noise(0, 1, 1)  # 添加高斯噪声
    add_noise.pepper_salt_noise(0.2)  # 添加椒盐噪声
    cv2.imshow('source', img)
    cv2.imshow('gauss noise', add_noise.img_gauss)
    cv2.imshow('pepper salt noise', add_noise.img_pepsalt)
    cv2.waitKey(0)

# -*- coding: utf-8 -*-

import numpy as np
import cv2
from numpy import shape
import random
from PIL import Image
from skimage import util
'''对图像加噪声'''

class Noise(object):

    def __init__(self, img):
        self.img=img
        self.img1=img1
        self.percetage = percetage
        self.Gimg=None
        self.SPimg=None



    # ---------私有函数------------
    # 高斯噪声
    def _Gaussian(self,means,sigma):
        self.Gimg = self.img
        self.Num_Gaussian = int(self.percetage * self.img.shape[0] * self.img.shape[1])
        for i in range(self.Num_Gaussian):
            rand_x = random.randint(0, self.img.shape[0] - 1)
            rand_y = random.randint(0, self.img.shape[1] - 1)
            self.Gimg[ rand_x, rand_y] = self.img[rand_x, rand_y] + random.gauss(means, sigma)
            if self.Gimg[rand_x, rand_y] < 0:
                self.Gimg[rand_x, rand_y] = 0
            elif self.Gimg[rand_x, rand_y] > 255:
                self.Gimg[rand_x, rand_y] = 255
        return self.Gimg

    # 椒盐噪声
    def _Saltpepper(self):
        self.SPimg=self.img
        Num_Gaussian = int(self.percetage * self.img.shape[0] * self.img.shape[1])
        for i in range(Num_Gaussian):
            rand_x = random.randint(0, self.img.shape[0] - 1)
            rand_y = random.randint(0, self.img.shape[1] - 1)
            if random.random()<0:
                self.SPimg[rand_x, rand_y] = 0
            elif random.random()>0:
                self.SPimg[rand_x, rand_y] = 255
        return self.SPimg

    # 调用噪声函数
    def _noise(self):
        self.noise_img = util.random_noise(self.img1, mode='poisson')
        return self.noise_img

    # -------导出接口-----

    # 实现高斯噪声功能
    def _MyGaussian(self):
        self._Gaussian(0,5)
        cv2.imshow('Gaussiannoise-img',self.Gimg)
        # cv2.imshow("Gaussiannoise-img", np.hstack([img, self.Gimg]))
    # 实现椒盐噪声功能
    def _MySaltpepper(self):
        self._Saltpepper()
        cv2.imshow('Salt&pepper-img', self.SPimg)
    # 实现噪声功能
    def _Mynoise(self):
        self._noise()
        cv2.imshow('noise-img', self.noise_img)


'''应用噪声'''
if __name__ == "__main__":
    img=cv2.imread('lenna.png',0)
    img1 = cv2.imread("lenna.png",1)

    cv2.imshow('src-img',img)
    percetage = 0.5

    N_img=Noise(img)
    # 实现高斯噪声功能
    N_img._MyGaussian()

    # 实现椒盐噪声功能
    N_img._MySaltpepper()

    # 实现噪声功能
    N_img._Mynoise()
    cv2.waitKey(0)
    cv2.destroyAllWindows()




import numpy as np
import cv2
from numpy import shape
import random
from skimage import util
def GaussianNoise(img,means,sigma,percetage):
    Noiseimg = img
    Noisenum = int(percetage*img.shape[0]*img.shape[1])
    for i in range(Noisenum):
        # 遍历预定次数，每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        rand_x = random.randint(0,img.shape[0]-1)
        rand_y = random.randint(0,img.shape[0]-1)
        #原始图片上加上高斯随机噪声
        Noiseimg[rand_x,rand_y] = Noiseimg[rand_x,rand_y] + random.gauss(means,sigma)
        # 记得限定范围，若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if Noiseimg[rand_x,rand_y] < 0:
            Noiseimg[rand_x,rand_y] = 0
        elif Noiseimg[rand_x,rand_y] > 255:
            Noiseimg[rand_x,rand_y] = 255

    return Noiseimg



img = cv2.imread('lenna.png',0)
gauss_img = GaussianNoise(img,2,4,0.9)
cv2.imshow('source',img)
cv2.imshow('lenna_GaussianNoise',gauss_img)
#接口生成gauss
noisr_gs_img = util.random_noise(img,mode = 'gaussian')
cv2.imshow('lenna_GaussianNoise2',noisr_gs_img)

cv2.waitKey(0)

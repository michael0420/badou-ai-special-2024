import numpy as np
import cv2
from numpy import shape
import random
from skimage import util
def SPNoise(SRC,percetage):
    Noiseimg=SRC
    Noisenum = int(percetage*SRC.shape[0]*SRC.shape[1])
    for i in range(Noisenum):
        # 遍历预定次数，每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        rand_x = random.randint(0,SRC.shape[0]-1)
        rand_y = random.randint(0,SRC.shape[0]-1)
        #随机判定0.5的概率赋值0或者255
        if random.random() <= 0.5:
            Noiseimg[rand_x,rand_y] = 0
        elif random.random() > 0.5:
            Noiseimg[rand_x,rand_y] = 255

    return Noiseimg
img = cv2.imread('lenna.png',0)
cv2.imshow('source',img)
gauss_img = SPNoise(img,0.6)
cv2.imshow('lenna_GaussianNoise',gauss_img)
#接口生成s&p
noisr_sp_img = util.random_noise(img,mode = 's&p')
cv2.imshow('lenna_GaussianNoise2',noisr_sp_img)

cv2.waitKey(0)

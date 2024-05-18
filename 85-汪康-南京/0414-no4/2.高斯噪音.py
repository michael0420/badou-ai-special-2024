# -*- coding: utf-8 -*-
'''@Time: 2024/4/16 21:29
高斯噪音
'''
import random
import numpy as np
import cv2
from PIL import Image
from skimage import util
#1、使用random.normal
image = cv2.imread("../lenna.png",0)
mean = 2
var = 0.8
sigma = 4
gaussian = np.random.normal(mean,sigma,image.shape).astype('uint8')
noisy_image = cv2.add(image,gaussian)
cv2.imshow('Noisy Image',np.hstack([image,noisy_image]))
cv2.waitKey(0)
cv2.destroyAllWindows()

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
		#每次取一个随机点
		#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
		#高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('../lenna.png',0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('../lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('',img2)
cv2.imshow('source lenna_GaussianNoise',np.hstack([img2,img1]))
cv2.waitKey(0)

#3 使用util.random_noise
img3 = cv2.imread("../lenna.png")
noise_gs_img=util.random_noise(img3)
cv2.imshow('',img3)
cv2.imshow('source poisson',noise_gs_img)
cv2.waitKey()
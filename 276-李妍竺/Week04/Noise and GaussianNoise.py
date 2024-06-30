'''
#调用噪声（直接）
import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util

img = cv.imread('lenna.png')
noise_poi = util.random_noise(img,mode='poisson')

cv.imshow('source',img)
cv.imshow('noise_img',noise_poi)
cv.waitKey()
cv.destroyAllWindows()
'''

#高斯噪声过程
import cv2
import numpy as np
from numpy import shape
import random

def GaussianNoise(src,mean,sigma,percentage):
    Noise_img = src
    Noise_num = int(percentage*src.shape[0]*src.shape[1])
    for i in range(Noise_num):
        rand_x = random.randint(0, src.shape[0] - 1) #高斯噪声图片边缘不处理，所以-1
        rand_y = random.randint(0, src.shape[1] - 1)
        #将生成的随机数放缩在[0,255]之间
        Noise_img[rand_x,rand_y] = Noise_img[rand_x,rand_y]+random.gauss(mean,sigma)
        #控制随机值不超出范围

        if Noise_img[rand_x, rand_y]<0:
            Noise_img[rand_x, rand_y]=0
        elif Noise_img[rand_x, rand_y]>255:
            Noise_img[rand_x, rand_y]=255
    return Noise_img

img = cv2.imread('lenna.png',0)
img1 = GaussianNoise(img,3,5,0.9)
#img1运行后，img改变，故想要输出对比图，需要img重新运行.
img = cv2.imread('lenna.png',0)
cv2.imshow('source',img)
cv2.imshow('Gaussian_noise',img1)
cv2.waitKey()

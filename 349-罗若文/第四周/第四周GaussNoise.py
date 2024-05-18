'''
给照片随机加上高斯噪声
'''
import cv2
import random
from skimage import util


# 生成高斯噪声的方法 src:图片路径 means,sigma：生成高斯随机数的两个参数，percetage:百分比
def gaussianNoise(rawImg, means, sigma, percent):
    # 需要噪声的图片直接等于cv2.imread后的图片
    noiseImg = rawImg
    #用百分比乘原图的长宽，得到需要加噪声的像素点数量
    noiseNum = int(percent * rawImg.shape[0] * rawImg.shape[1])

    for i in range(noiseNum):
        # random.randint生成随机数

        randX = random.randint(0, rawImg.shape[0] - 1) # 获得0到原图横轴-1的随机数 高斯噪声图片边缘不做处理，所以-1
        randY = random.randint(0, rawImg.shape[1] - 1) # 获得0到原图纵轴-1的随机数 高斯噪声图片边缘不做处理，所以-1
        # 此处在原有像素灰度值上加上随机数
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        # 若灰度值小于0 的则强制为0，若灰度值大于255则强制255
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg


img = cv2.imread('lenna.png', 0)  #直接读取灰度图
img1 = gaussianNoise(img, 2, 4, 0.8) #灰度图加高斯噪声

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度图

#使用工具类转高斯噪声
guassianImg = util.random_noise(img, mode='gaussian')  #三通道加噪声

cv2.imshow('gray_gaussian', img1)
cv2.imshow('raw_gray', img2)
cv2.imshow('auto_gaussin', guassianImg)
cv2.waitKey(0)
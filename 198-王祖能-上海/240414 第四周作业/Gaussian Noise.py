'''
高斯噪声Gaussian Noise加强
'''
import numpy
import random
import cv2
import numpy as np


def Gaussian_Noise1(src, mu, sigma, percentage):
    '''`
    :param src:读取原始图片
    :param mu:高斯分布均值
    :param sigma:高斯分布标准差
    :param percentage:进行高斯噪声加强的图像比例0~1
    :return:返回加强后的图片
    '''
    h, w = src.shape
    dst = src.copy()  # 直接赋值src后续return会修改src的值
    total = int(h * w * percentage)  # 确定需要高斯化的像素点数
    # total = int(src.shape[0] * src.shape[1] * percentage)  # 直接调用计算，省略赋值过程
    for i in range(total):  # 随机化会有个别点重复，对结果基本不影响，不需要去重
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        m = dst[x, y] + random.gauss(mu, sigma)  # 先对临时变量m赋值，判断取值范围后再将值赋给图片
        if m < 0:
            m = 0
        elif m > 255:
            m = 255
        dst[x, y] = m
    return dst


def Gaussian_Noise2(src, mu, sigma, percentage):
    h, w = src.shape
    dst = src.copy()  # 否则直接赋值src后续return会修改src的值
    total = int(h * w * percentage)  # 确定需要高斯化的像素点数
    for i in range(total):
        x = random.randint(0, h - 1)
        y = random.randint(0, w - 1)
        dst[x, y] = dst[x, y] + random.gauss(mu, sigma)
        '''
        直接对图片取值计算当超过0~255范围后，会自动循环处理，比如计算值260，输出图片值为4，结果偏差大
        '''
        if dst[x, y] < 0:
            dst[x, y] = 0
        elif dst[x, y] > 255:
            dst[x, y] = 255
    return dst


img = cv2.imread('lenna.png', 0)
img_Gaussian1 = Gaussian_Noise1(img, 50, 10, 0.5)  # 正确方法
img_Gaussian2 = Gaussian_Noise2(img, 50, 10, 0.5)
print(img_Gaussian1[0] == img_Gaussian2[0])  # 验证结果存在不同
cv2.namedWindow('Gaussian_Noise_Img', cv2.WINDOW_NORMAL)
cv2.imshow('Gaussian_Noise_Img', np.hstack([img, img_Gaussian1, img_Gaussian2]))
cv2.waitKey()

# 彩色图片分离通道进行高斯加噪，各通道按照不同的高斯噪声  # 也可以对通道进行循环，但只能统一参数mu，sigma
img = cv2.imread('lenna.png')
b, g, r = cv2.split(img)
bGauss, gGauss, rGauss = Gaussian_Noise1(b, 50, 10, 0.5), Gaussian_Noise1(g, 50, 10, 0.5), Gaussian_Noise1(r, 50, 10, 0.5)
img_color_Gaussian = cv2.merge((bGauss, gGauss, rGauss))
cv2.namedWindow('ColorBase', cv2.WINDOW_NORMAL)
cv2.imshow('ColorBase', np.hstack([bGauss, gGauss, rGauss]))
cv2.waitKey()
cv2.imshow('ColorBase', np.hstack([img, img_color_Gaussian]))
cv2.waitKey()
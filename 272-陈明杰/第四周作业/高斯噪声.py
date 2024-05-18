import cv2
import numpy as np
import random


# 添加高斯噪声
def gaussianNoise(src: np.ndarray, means, sigma, percetage):
    high, width = img.shape[:2]
    NoiseImg = src.copy()
    # 取percetage比例的像素点进行添加高斯噪声
    NoiseNum = int(percetage * high * width)
    for k in range(NoiseNum):
        # 在图像[0, high - 1]区间中生成随机下标，对其进行添加高斯噪声
        i = random.randint(0, high - 1)
        # 在图像[0, width - 1]区间中生成随机下标，对其进行添加高斯噪声
        j = random.randint(0, width - 1)
        # random.gauss(means, sigma)生成符合高斯分布的数值
        tmp = NoiseImg[i, j] + random.gauss(means, sigma)
        if tmp > 255:
            tmp = 255
        elif tmp < 0:
            tmp = 0
        NoiseImg[i, j] = tmp
    return NoiseImg


img = cv2.imread("lenna.png",0)
# 添加高斯噪声
dst = gaussianNoise(img, means=0, sigma=1, percetage=1)
cv2.imshow("src", img)
cv2.imshow("dst", dst)
cv2.waitKey(0)

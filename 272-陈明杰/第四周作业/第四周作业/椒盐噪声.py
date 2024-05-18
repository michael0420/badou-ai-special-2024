import cv2
import numpy as np
import random


def pepperSaltNoise(img: np.ndarray, SNR):
    high, width = img.shape[:2]
    # 注意这里要创建一个拷贝，不能直接赋值，直接赋值=引用
    NoiseImg = img.copy()
    # 取percetage比例的像素点进行添加椒盐噪声
    NoiseNum = int(SNR * high * width)
    for k in range(NoiseNum):
        # 在图像[0, high - 1]区间中生成随机下标，对其进行添加椒盐噪声
        i = random.randint(0, high - 1)
        # 在图像[0, width - 1]区间中生成随机下标，对其进行添加椒盐噪声
        j = random.randint(0, width - 1)
        # 椒就是黑，盐就是白，所以按随机数把对应位置的像素值改为0/255
        if random.random() <= 0.5:
            NoiseImg[i, j] = 255
        else:
            NoiseImg[i, j] = 0
    return NoiseImg


src = cv2.imread("lenna.png", 0)
# 添加椒盐噪声
dst = pepperSaltNoise(src, SNR=0.2)
cv2.imshow("src", src)
cv2.imshow("dst", dst)
print(src)
print(dst)
cv2.waitKey(0)

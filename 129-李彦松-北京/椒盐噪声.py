import numpy as np
import cv2
import random

def SaltAndPepper_optimized(src, SNR):
    # 将图像转换为 float64 类型
    NoiseImg = src.astype(np.float64)

    # 计算总像素数目 SP
    SP = src.shape[0] * src.shape[1]
    # 得到要加噪的像素数目 NP
    NP = int(SP * SNR)

    # 随机获取要加噪的每个像素位置
    randX = np.random.randint(0, src.shape[0]-1, size=NP)
    randY = np.random.randint(0, src.shape[1]-1, size=NP)

    # 指定像素值为255或者0
    noise = np.random.choice([0, 255], size=NP)

    # 将噪声添加到图像上
    NoiseImg[randX, randY] = noise

    # 将图像转换回 uint8 类型
    NoiseImg = NoiseImg.astype(np.uint8)

    return NoiseImg

img = cv2.imread('lenna.png',0)
cv2.imshow('source',img)
img1 = SaltAndPepper_optimized(img,0.8)
cv2.imshow('lenna_SaltAndPepper', img1)
cv2.waitKey(0)
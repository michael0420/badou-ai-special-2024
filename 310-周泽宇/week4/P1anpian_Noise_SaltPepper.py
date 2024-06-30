import numpy as np
import random
import cv2

def salt_pepper_noise(img, ratio):
    # ratio是加噪的比例
    output = np.zeros(img.shape, np.uint8)
    # 使用2重for循环遍历每一个像素点进行加噪(包括边缘)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 产生一个0-1的随机数 用于控制加噪点的数量(判断是否加噪)
            rand = random.random()
            if rand < ratio: # 随机数 < 椒盐噪声的比例 则加噪
                if random.random() > 0.5: # 等比例随机产生0和255噪声
                    output[i][j] = 255
                else:
                    output[i][j] = 0
            else:
                output[i][j] = img[i][j]
    return output

def salt_pepper_noise2(img, ratio):
    # ratio是加噪的比例
    output = img
    # 应加噪的像素点数量为原图数量*比例
    NoiseNum = int(img.shape[0] * img.shape[1] * ratio)
    # 随机选取点并直接加噪 缺点是可能会选取到重复的点
    for i in range(NoiseNum):
        # randX Y分别代表随机的X,Y像素点坐标 img.shape[0]-1代表默认边缘不会加噪音
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        # 只需要随机决定0或255噪音 不需要再判断是否加噪
        if random.random() <= 0.5:
            output[randX, randY] = 0
        else:
            output[randX, randY] = 255
    return output

'''test:'''
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('input',img2)
out1 = salt_pepper_noise(img2, 0.6)
out2 = salt_pepper_noise2(img2, 0.6)
cv2.imshow("output1", out1)
cv2.imshow("output2", out2)
cv2.waitKey(0)
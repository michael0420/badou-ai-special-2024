import cv2
import numpy as np
import random


def salt_noise(imgsrc, percent):
    noiseImg = imgsrc
    print("imgsrc.shape:", imgsrc.shape[0], imgsrc.shape[1])
    noiseNumber = int(percent * imgsrc.shape[0] * imgsrc.shape[1])

    for i in range(noiseNumber):
        # 随机选择一个像素点
        # 图片边缘不处理，故-1
        randX = np.random.randint(0, imgsrc.shape[0] - 1)
        randY = np.random.randint(0, imgsrc.shape[1] - 1)

        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
        if random.random() < 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg


# 读取图片
img = cv2.imread('lenna.png', 0)
# 显示图
cv2.imshow("original image", img)
# 添加椒盐噪声
img_salt = salt_noise(img, 0.8)
cv2.imshow("salt noise image", img_salt)
cv2.waitKey(0)
cv2.destroyAllWindows()

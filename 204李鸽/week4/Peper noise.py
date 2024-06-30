import cv2
import numpy as np
import random
def peper_noise(img, percentage):
    noise_img = img.copy()
    h, w, _ = img.shape
    noise_num = int(percentage * h * w)
    for i in range(noise_num):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        randX = random.randint(0, h - 1)
        randY = random.randint(0, w - 1)
        # random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0，
        # 只是加一个概率，就像抛硬币
        if random.random() <= 0.5:
            noise_img[randX, randY] = 0
        else:
            noise_img[randX, randY] = 255
    return noise_img

img = cv2.imread('face.jpg')
noise_image = peper_noise(img, 0.2)
cv2.imshow('src', img)
cv2.imshow('noise', noise_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
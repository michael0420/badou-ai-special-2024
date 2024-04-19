import cv2
import random


def gaoshizaosheng(src_img, param, param1, param2):
    src = src_img  # 将原图赋值给新变量
    process_number = int(src_img.shape[0] * src_img.shape[1] * param2)
    for i in range(process_number):
        # 获取随机坐标
        random_x = random.randint(0, src_img.shape[0] - 1)
        random_y = random.randint(0, src_img.shape[1] - 1)
        src[random_x, random_y] = src[random_x, random_y] + random.gauss(param, param1)
        if src[random_x, random_y] < 0:
            src[random_x, random_y] = 0
        elif src[random_x, random_y] > 255:
            src[random_x, random_y] = 255
    return src


# 将图片读成单通道
src_img = cv2.imread('lenna.png', 0)
gaoshi_img = gaoshizaosheng(src_img, 3, 5, 0.6)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise', gaoshi_img)
cv2.waitKey(0)

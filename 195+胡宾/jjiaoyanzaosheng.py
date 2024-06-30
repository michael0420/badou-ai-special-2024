import cv2
import random


def jiaoyan_zaosheng(src_img, param2):
    src = src_img  # 将原图赋值给新变量
    process_number = int(src_img.shape[0] * src_img.shape[1] * param2)
    for i in range(process_number):
        # 获取随机坐标
        random_x = random.randint(0, src_img.shape[0] - 1)
        random_y = random.randint(0, src_img.shape[1] - 1)
        if random.random() >= 0.5:
            src[random_x, random_y] = 0
        else:
            src[random_x, random_y] = 255
    return src


# 将图片读成单通道
src_img = cv2.imread('lenna.png', 0)
jiaoyan_img = jiaoyan_zaosheng(src_img, 0.2)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_jiaoyan', jiaoyan_img)
cv2.waitKey(0)

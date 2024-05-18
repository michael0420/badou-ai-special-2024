import numpy as np
import cv2
from numpy import shape
import random

def SaltPepperNoise(src,percetage):
    # 获取需要需要循环的像素数量
    per_pixel = int(percetage * src.shape[0] * src.shape[1])
    # 遍历像素数量
    for i in range(per_pixel):
        # 获取横、纵坐标，-1是因为坐标是从0开始
        x = random.randint(0, src.shape[0]-1)
        y = random.randint(0, src.shape[1]-1)
        # 生产随机数，50%的可能把元素值变为255或0
        if random.randint(0, 100) >= 50:
            src[x, y] = 255
        else:
            src[x, y] = 0

    return src

# 获取lenna的灰度图片
img = cv2.imread("lenna.png",0)

spn_img = SaltPepperNoise(img,0.6)
img3 = cv2.imread('lenna.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

cv2.imshow("spnoise",spn_img)
cv2.imshow('img3',img3)
cv2.waitKey(0)


# 调库方法，可以直接添加噪声
# noise_gs_img=util.random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs)
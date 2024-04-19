# coding = utf-8

'''
    实现椒盐噪声
'''

import numpy as np
import  cv2
import random
from skimage import util as ski


img = cv2.imread('lenna.png')
cv2.imshow('lenna', img)


'''
使用接口实现椒盐噪声
'''
s_p = ski.random_noise(img, mode='s&p')
cv2.imshow('s&p', s_p)


'''
手动实现椒盐噪声
'''
def sp(img, perc):
    noise_img = img
    noise_num = int(img.shape[0] * img.shape[1] * perc)
    # print(np.shape(noise_img))

    # 遍历3通道
    for j in range(noise_img.shape[2]):
        # 遍历噪声点出现的随机坐标
        for i in range(noise_num):
            # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
            # random.randint()生成随机整数
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)

            # random.random()生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
            if random.random() < 0.5:
                noise_img[randX, randY, j] = 0
            elif random.random() > 0.5:
                noise_img[randX, randY, j] = 255
    return noise_img


src = cv2.imread('lenna.png')
sp = sp(src, 0.3)
cv2.imshow('s&pImg', sp)
cv2.waitKey(0)

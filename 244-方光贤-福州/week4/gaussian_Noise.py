import cv2
import random

def gaussian_Noise(ori_img, mean, sigma, percetage):
    #定义高斯噪声 参数包括原图 均值 方差和百分比
    noisy_img = ori_img
    #定义加噪像素点个数
    NoiseNum = int(percetage * ori_img.shape[0] * ori_img.shape[1])
    #生成随机行和随即列定点添加噪声的点 由于元素量较多 生成的重复点可以忽略不计
    #rand_x代表随机行 rand_y代表随机列 randint代表随机整数
    #shape[0] shape[1]分别代表行和列
    for i in range(NoiseNum):
        rand_x = random.randint(0, ori_img.shape[0] - 1)
        rand_y = random.randint(0, ori_img.shape[1] - 1)
        #添加高斯随机数
        noisy_img[rand_x, rand_y] = ori_img[rand_x, rand_y] + random.gauss(mean, sigma)
        #防止越界 要将超出255和0的值恢复到0-255的范围内
        if noisy_img[rand_x, rand_y] < 0:
            noisy_img[rand_x, rand_y] = 0
        elif noisy_img[rand_x, rand_y] > 255:
            noisy_img[rand_x, rand_y] = 255

    return noisy_img

img = cv2.imread("lenna.png", 0)
noisy_img = gaussian_Noise(img, 3, 3, 0.9)
img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', gray_img)
cv2.imshow('lenna_GaussianNoise', noisy_img)
cv2.waitKey(0)
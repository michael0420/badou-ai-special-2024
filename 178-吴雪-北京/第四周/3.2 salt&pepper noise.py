"""
椒盐噪声： salt & pepper noise
    给一副数字图像加上椒盐噪声的处理顺序：
    1、指定信噪比SNR(信号和噪声所占比例)，其取值范围在[0,1]之间
    2、计算总像素数目SP=h*w，得到要加噪的像素数目：NP = SP * SNR
    3、随机获取要加噪的每个像素位置P(i,j)
    4、指定像素值为255或者0
    5、重复3、4两个步骤完成所有NP个像素的加噪
"""
import cv2
import random


def Salt_Pepper_Noise(src, percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(3):
        for j in range(NoiseNum):
            rand_x = random.randint(0, src.shape[0] - 1)
            rand_y = random.randint(0, src.shape[1] - 1)
            if random.random() < 0.5:
                NoiseImg[rand_x, rand_y] = 0
            else:
                NoiseImg[rand_x, rand_y] = 255
    return NoiseImg


img = cv2.imread('E:/Desktop/jianli/lenna.png')
dst = Salt_Pepper_Noise(img, 0.01)
cv2.imshow('Source', img)
cv2.imshow('Salt&Pepper Noise', dst)
# cv2.imwrite("lenna_noise.png", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
高斯噪声：GaussianNoise
    一个正常的高斯采样分布公式，得到输出像素Pout
        Pout = Pin + random.gauss(means, sigma)
    其中random.gauss是通过sigma和mean来生成符合高斯分布的随机数。
给一副数字图像加上高斯噪声的处理顺序如下：
1、指定输入参数means, sigma， percentage(百分比=信噪比=SNR)
2、计算总像素数目SP=h*w，得到要加噪的像素数目：NP = SP * SNR
3、生成高斯随机数，根据(上面公式)输入像素计算出输出像素
4、重新将像素值放缩在[0, 255]之间
5、循环所有需要做高斯噪声的像素NP
6、输出像素
"""
import cv2
import random


def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        NoiseImg[rand_x, rand_y] = NoiseImg[rand_x, rand_y] + random.gauss(means, sigma)
        if NoiseImg[rand_x, rand_y] < 0:
            NoiseImg[rand_x, rand_y] = 0
        elif NoiseImg[rand_x, rand_y] > 255:
            NoiseImg[rand_x, rand_y] = 255
    return NoiseImg


img = cv2.imread("E:/Desktop/jianli/lenna.png", 0)
dst = GaussianNoise(img, 2, 4, 0.95)
cv2.imshow("source", img)
cv2.imshow("Gaussian noise", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

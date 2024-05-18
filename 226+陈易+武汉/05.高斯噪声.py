import cv2
import random

def GaussNoise(src,means,sigma,percetage):
    NoiseImg = src                                              # 拿到要处理的图片
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])     # 计算要处理的像素点数
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX代表随机生成的行，ranY代表随机生成的列
        # 高斯噪声图片边缘不处理，故-1
        # random.randint()生成随机整数
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        # 在原像素值灰度值上加价随机数
        NoiseImg[randX,randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)
        # 如果灰度值小于0则强制为0，如果灰度值大于255则强制为255
        if NoiseImg[randX,randY] < 0:
            NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY] > 255:
            NoiseImg[randX,randY] = 255
    return NoiseImg
img = cv2.imread("lenna.png",0)
img_guass = GaussNoise(img,2,4,0.8)             # 高斯噪声图
img = cv2.imread('lenna.png')                   # 原图
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度图
cv2.imshow("img",img)
cv2.imshow("img_guass",img_guass)
cv2.imshow("img2",img2)
cv2.waitKey()
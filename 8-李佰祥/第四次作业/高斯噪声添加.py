import cv2
import numpy
import random
#添加高斯噪声流程
#1.获得参数sigma和mean
#2.生成高斯随机数
#3.根据输入的像素结合高斯随机数计算结果像素值
#有可能结合高斯随机数后，像素值大于255，所以需要将像素值缩放在0-255
#4.循环处理所有或部分像素值
img = cv2.imread("../../lenna.png",0)
sigma = 10
mean = 5
percetage = 0.8


noiseNUM = int(img.shape[0] * percetage * img.shape[1])
noiseImg = img
for i  in range(noiseNUM):
    #每次取随机点
    randX = random.randint(0,img.shape[0]-1)
    randY = random.randint(0, img.shape[1] - 1)
    #给像素点加上噪音
    noiseImg[randX,randY] = noiseImg[randX,randY] + random.gauss(mean,sigma)

    if noiseImg[randX,randY] <0:
        noiseImg[randX, randY]=0
    elif noiseImg[randX,randY] >255:
        noiseImg[randX, randY] = 255

cv2.imshow("noiseImg",noiseImg)
cv2.waitKey(0)









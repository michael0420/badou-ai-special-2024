import cv2
import random

def GsNoise(src,percetage):
    newImage = src
    num = int(percetage*src.shape[0]*src.shape[1])
    for i in range(num):
        # 随机区图片的像素坐标
        x = random.randint(0,src.shape[0]-1);
        y = random.randint(0,src.shape[1]-1);
        if random.random() < 0.5:
            newImage[x, y] = 255
        else:
            newImage[x, y] = 0
    return newImage

img = cv2.imread('../lenna.png',0)
img1 = GsNoise(img,0.5)
img = cv2.imread('../lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('Gs',img1)
cv2.waitKey(0)


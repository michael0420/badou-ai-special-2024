import cv2
import numpy as np
import random

class Noise:
    def __init__(self):
        pass

    def gauses(self,img,percent):

        changePoint = int(img.shape[0]*img.shape[1]*percent)
        h,w=img.shape[0],img.shape[1]

        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for i in range(changePoint):


            randX = random.randint(0,h-1)
            randY = random.randint(0,w-1)

            img_gray[randX][randY]= img_gray[randX][randY] + random.gauss(0,1)

            if img_gray[randX][randY]>255:
                img_gray[randX][randY]=255
            elif img_gray[randX][randY]<0:
                img_gray[randX][randY]=0

        return img_gray


    def pepperSalt(self, img, percent):
        changePoint = int(img.shape[0] * img.shape[1] * percent)
        h, w = img.shape[0], img.shape[1]

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i in range(changePoint):

            randX = random.randint(0, h - 1)
            randY = random.randint(0, w - 1)


            if (img_gray[randX][randY] / 255)>0.5:
                img_gray[randX][randY]=255
            else:
                img_gray[randX][randY]=0


            if img_gray[randX][randY] > 255:
                img_gray[randX][randY] = 255
            elif img_gray[randX][randY] < 0:
                img_gray[randX][randY] = 0

        return img_gray


if __name__ == '__main__':

    img=cv2.imread('lenna.png')
    nosie =Noise()
    nosie_img=nosie.gauses(img,0.8)
    nosie_img1=nosie.pepperSalt(img,0.2)
    img_gary=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('orgin',img_gary)

    cv2.imshow('nosie',nosie_img)
    cv2.imshow('pepersalt',nosie_img1)

    cv2.waitKey(0)


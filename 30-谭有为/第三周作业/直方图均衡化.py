import numpy as  np
import cv2
import matplotlib.pyplot as  plt

def img_gray(img):     #图像的灰度化
    heigt,width=img.shape[:2]
    gray_img=np.zeros((heigt,width),img.dtype)
    for i in  range(heigt):
       for j in range(width):
          m=img[i,j]
          gray_img[i,j]=int(m[0]*0.11 +m[1]*0.59 +m[2]*0.3)
    return gray_img

def img_equalization(img):   #灰度图像的均衡化
     equalization_img=cv2.equalizeHist(img)
     return  equalization_img

def color_img_equalization(img):  #彩色图片的均衡化
    (b,g,r)=cv2.split(img)
    bE=cv2.equalizeHist(b)
    gE=cv2.equalizeHist(g)
    rE=cv2.equalizeHist(r)
    color_equalization_img=cv2.merge((bE,gE,rE))
    return  color_equalization_img

if __name__=='__main__':
    img=cv2.imread('F:/PNG/lenna.png')
    cv2.imshow('Original image',img)
    img1=img_gray(img)
    cv2.imshow('Grayscale image',img1)
    img2=img_equalization(img1)
    cv2.imshow('equalization image',img2)
    img3=color_img_equalization(img)
    cv2.imshow('equalization color image',img3)
    cv2.waitKey()


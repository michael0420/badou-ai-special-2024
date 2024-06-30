import random as rd
import numpy as np
import cv2
import matplotlib.pyplot as  plt

def img_gauss_noise(img,mu,sigma,proportion):   #高斯噪声，高斯即正态分布，mu和sigma表示正态分布中的μ 和 σ,分别代表均值和标准差，proportion表示比例
    gauss_noise_img=img   #加噪后的图片先等于原图
    height,width=img.shape[:2]  #取原图的高和宽
    pixel_count=int(height*width*proportion)  #高度*宽度*比例 得到需要加噪的像素点数量
    for i in range(pixel_count):
        x=rd.randint(0,height-1)   #随机像素点的X坐标
        y=rd.randint(0,width-1)    #随机像素点的Y坐标
        gauss_noise_img[x,y]=gauss_noise_img[x,y]+rd.gauss(mu,sigma)   #像素点的坐标的像素值+随机生成的高斯值
        if gauss_noise_img[x,y]<0:
            gauss_noise_img[x,y]=0
        elif gauss_noise_img[x,y]>255:
              gauss_noise_img[x,y]=255   #保证像素值在0-255之间
    return gauss_noise_img

def img_salt_pepper_noise(img,proportion):   #椒盐噪声
    salt_pepper_noise_img=img
    height,width=img.shape[:2]
    pixel_count=int(height*width*proportion)
    for i in range(pixel_count):
        x=rd.randint(0,height-1)
        y=rd.randint(0,width-1)          #截止到这里，跟高斯噪声一样
        if rd.random()<0.5:   #random.random 函数，随机生成0-1之间的浮点数
            salt_pepper_noise_img[x,y]=0
        else:
            salt_pepper_noise_img[x,y]=255   # 像素值随机变为0或者变为255
    return salt_pepper_noise_img

def img_bgr2rgb(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)
    return img
def img_gray(img):   #图像灰度化
    img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
    return img
'''
if __name__=='__main__':
    img=cv2.imread('F:/PNG/lenna.png',0)
#cv2.imread（’图片路径‘，0）--读取灰度图片；1表示读取彩色图片，不包含alpha通道；-1表示读取彩色图片，包含alpha通道
    cv2.imshow('Original image',img)
    img1=img_gauss_noise(img,20,5,0.8)
    cv2.imshow('gauss_noise_img',img1)
    img2=img_salt_pepper_noise(img,0.2)
    cv2.imshow('salt_pepper_noise_img',img2)
    cv2.waitKey()

'''
if __name__=='__main__':
    plt.subplot(221)
    img0=cv2.imread('F:/PNG/lenna.png')
    plt.imshow(img_bgr2rgb(img0))
    plt.subplot(222)
    img=img_gray(img0)
    plt.imshow(img,cmap='gray')
    plt.subplot(223)
    img1=img_gauss_noise(img,20,5,0.8)
    plt.imshow(img1,cmap='gray')
    plt.subplot(224)
    img2=img_salt_pepper_noise(img,0.5)
    plt.imshow(img2,cmap='gray')
    plt.show()




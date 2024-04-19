import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#灰度处理
# img=cv2.imread('LYF.jpeg')
# h,w=img.shape[:2]
# img_gray=np.zeros([h,w],img.dtype)
# img_gray2=np.zeros([h,w],img.dtype)
# for i in range(h):
#     for j in range(w):
#         m=img[i,j]
#         img_gray[i,j]=int((m[0]+m[1]+m[2])/3)
#         img_gray2[i,j]=(m[0]*30+m[1]*59+m[2]*11)/100
# cv2.imshow('',img_gray)
# cv2.waitKey()
# cv2.imshow('',img_gray2)
# cv2.waitKey()
# img2 =plt.imread('LYF.jpeg')
# plt.subplot(121)
# plt.imshow(img2)
# plt.subplot(122)
# img_grey = rgb2gray(img2)
# plt.imshow(img_grey, cmap='gray')
# plt.show()


#最邻近插值
def a(img,w,h):
    height, width, channels = img.shape
    emptyImage = np.zeros((h, w, channels), np.uint8)
    sh = h / height
    sw = w / width
    for i in range(h):
        for j in range(w):
            x = int(i / sh +0.5)
            y = int(j / sw+0.5)
            if(y>=width):
                y=y-1
            if (x >= height):
                x = x - 1
            emptyImage[i, j] = img[x, y]
    return  emptyImage
img = cv2.imread('LYF.jpeg')
zoom = a(img,300,400)
cv2.imshow('', zoom)
cv2.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# from skimage.color import rgb2gray

img = cv2.imread("./lenna.png")
cv2.imshow("lenna.png", img)
h, w = img.shape[0:2]
empty_img = np.zeros((h, w), img.dtype)
#print(empty_img.shape)
# 灰度化  可以直接用from skimage.color import rgb2gray   img_gray = rgb2gray(img)
# 或img_gray = cv2.cvt2Color(img, cv2.COLOR_BGR2GRAY)
def img_gray(img):
    '''
    :param img: 传入图片
    :return: 输出灰度化后的图片
    '''
    for i in range(h):
        for j in range(w):
            empty_img[i,j] = int(img[i, j][0]*0.11 + img[i, j][1]*0.59 + img[i, j][2]*0.3) #BGR
    return empty_img
img_gray =  img_gray(img)
print(img_gray)
cv2.imshow('image_gray', img_gray)

# 二值图
image_blackwhite1 = np.where(img_gray/255 > 0.5, 1, 0)
print(image_blackwhite1.dtype) #int32型
# cv2.imshow('image_blackwhite', image_blackwhite1)   报错 CV无法显示int32型
def image_blackwhite(img_gray):
    '''
    :param img_gray: 传入灰度图
    :return: 输出二值图
    '''
    img_gray = img_gray/255  #灰度图归一化
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if img_gray[i, j] > 0.5:
                img_gray[i, j] = 1
            else:
                img_gray[i, j] = 0
    return img_gray
image_blackwhite = image_blackwhite(img_gray)
print(image_blackwhite.dtype) #float64型
print(image_blackwhite1 == image_blackwhite)
cv2.imshow('image_blackwhite', image_blackwhite)

# 用plt展示图片
plt.subplot(221)
img = plt.imread("./lenna.png")
plt.imshow(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.subplot(223)
plt.imshow(image_blackwhite, cmap='gray')
plt.subplot(224)
plt.imshow(image_blackwhite1, cmap='gray')   # CV无法显示int32型，但是plt显示正常
plt.show()


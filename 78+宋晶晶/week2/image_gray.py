import os
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

HERE = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(HERE)
FILE_PATH = os.path.join(PARENT_DIR,'src','lenna.png')

'''Gray'''
img = cv2.imread(FILE_PATH)
# img.shape is a tuple, (height,width,array size)
h,w = img.shape[:2]
# img.dtype is unit8, which is a type of store image, limite the numpy array range of 0-255
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        # related float method translate img to gray mode
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)


# cv2.imshow("image show gray",img_gray)
cv2.imshow('image show gray', img_gray)
cv2.waitKey(0)  # 等待任意键按下，0表示无限等待
cv2.destroyAllWindows()  # 销毁所有窗口
# create a 2*2 canvas, this img shows in the first position
plt.subplot(221)
plt.imshow(img)

# create a 2*2 canvas, this img_gray shows in the second position
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

'''another method to create image gray'''
img_gray_2 = rgb2gray(img)
plt.subplot(223)
plt.imshow(img_gray_2,cmap='gray')

'''Binarization'''
# rows,cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if img_gray[i,j]<=0.5:
#             img_gray_2[i,j]=0
#         else:
#             img_gray_2[i,j]=1
           
img_binary = np.where(img_gray_2 >=0.5,1,0)
plt.subplot(224)
plt.imshow(img_binary,cmap="gray")
# if can not show image, can add a plt.show()
plt.show()
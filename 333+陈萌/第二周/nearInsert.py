import cv2
import numpy as np
from matplotlib import pyplot as plt

# img = plt.imread('lenna.png')
img = cv2.imread('lenna.png')
print(img.shape)

# 对图片进行扩大
def largeImg(img,scale):
    # 获取图片旧的宽、高
    h_old,w_old = img.shape[:2]
    # 获取图片新的宽、高
    h_new,w_new = scale[:2]

    # 计算比例
    h_rate = h_new/h_old
    w_rate = w_new/w_old

    newImg = np.zeros((h_new,w_new,3),dtype=img.dtype)
    for i in range(h_new):
        for j in range(w_new):
            # 计算原图的坐标
            # 如果 x_new - x 小于 0.5 取 x 否则取 x+1
            x = int(i/h_rate+0.5)
            y = int(j/w_rate+0.5)
            newImg[i,j] = img[x,y]
    return newImg


# plt.subplot(2,2,1)
# plt.imshow(img)
# plt.subplot(2,2,2)
# # 等比例放大缩小
# plt.imshow(largeImg(img,(800,800)))
# plt.subplot(2,2,3)
# plt.imshow(largeImg(img,(400,400)))
# plt.show()

cv2.imshow('img',img)
cv2.imshow('large',largeImg(img,(800,800)))
cv2.imshow('small',largeImg(img,(200,200)))
cv2.imshow('other',largeImg(img,(700,200)))
cv2.waitKey(0)
cv2.destroyAllWindows()

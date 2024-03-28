import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#使用plt读取原图片
org_img = plt.imread("../../lenna.png")

#第一个位置显示原图
plt.subplot(121)
plt.imshow(org_img)
#第一个位置显示二值图
plt.subplot(122)
gray_img = rgb2gray(org_img) #使用rgb2gray方法直接转换为灰度图
binary_img = np.where(gray_img>=0.5,1,0)
plt.imshow(binary_img,cmap='gray')

plt.show()












"""
@author: zhuhong

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
def gray(img):
    h, w = img[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
    return img_gray

img = cv2.imread("lenna.png")

# 进行归一化的灰度化
img_gray = rgb2gray(img)
# 二值化
img_binary = np.where(img_gray >= 0.5,1,0)

plt.subplot(131)
# cv2显示是brg，用plt展示的话需要转换成rgb
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(132)
plt.imshow(img_gray, cmap='gray')
plt.title('Gray Image')

plt.subplot(133)
plt.imshow(img_binary,cmap='gray')
plt.title('Binary Image')

plt.show()  # 显示图像

cv2.waitKey(0)
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 手工灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray_01 = np.zeros([h,w],dtype=img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray_01[i,j] = int(m[0]*0.11 + m[1]*0.59 +m[2]*0.3)
# cv2.imshow("image show gray 01",img_gray_01)


# 调接口灰度化
img_gray_02 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image show gray 02",img_gray_02)

# 手工二值化
img_gray_binary_01 = rgb2gray(img)
row,col= img_gray_binary_01.shape
for i in range(row):
    for j in range(col):
        if img_gray_binary_01[i,j] >= 0.5:
            img_gray_binary_01[i,j] = 1
        else:
            img_gray_binary_01[i,j] = 0
# cv2.imshow("image show binary 01",img_gray_binary_01)
# cv2.waitKey(0)
# 调接口二值化
img_gray = rgb2gray(img)
img_gray_binary_02 = np.where(img_gray >= 0.5, 1, 0)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False
plt.subplot(321)
plt.title("原图")
orgimg = plt.imread("lenna.png") 
plt.imshow(orgimg)
plt.subplot(323)
plt.title("手工灰度图")
plt.imshow(img_gray_01, cmap='gray')
plt.subplot(324)
plt.title("调接口灰度图")
plt.imshow(img_gray_02, cmap='gray') 
plt.subplot(325)
plt.title("手工二值图")
plt.imshow(img_gray_binary_01, cmap='gray')
plt.subplot(326)
plt.title("调接口二值图")
plt.imshow(img_gray_binary_02, cmap='gray')
plt.show()
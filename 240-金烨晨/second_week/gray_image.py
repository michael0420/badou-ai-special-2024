import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# img = plt.imread("../lenna.png")  ##读取图片信息，获取图片高度，宽度和通道数
img = cv2.imread("../lenna.png")    # bgr

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img)

img = cv2.imread("../lenna.png")

# 灰度图
plt.subplot(222)
h, w = img.shape[:2]    ##获取高度和宽度
img_gray = np.zeros([h,w],img.dtype)    ##创建一张img同大小同类型全是0的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]    #获取当前高宽的BGR坐标
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
plt.imshow(img_gray,cmap='gray')    #cmap设置颜色为grap

# 灰度图
plt.subplot(223)
# img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_gray = rgb2gray(img)
plt.imshow(img_gray,cmap="gray")

# 二值图
plt.subplot(224)
img_binary = np.where(img_gray>=0.5, 1, 0)
plt.imshow(img_binary, cmap='gray')
plt.show()

"""
@author 193-杨婷
彩色图像灰度化处理
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# 手撸灰度化
img = cv2.imread('lenna.png')
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 此处为BGR坐标（opencv读进来的图片通道顺序是BGR）
        img_gray[i, j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)  # gray=0.3R+0.59G+0.11B
print("original image%s" % img)
print("image show gray:%s" % img_gray)
# 无法直接将OpenCV的图像显示函数和Matplotlib的子图创建函数混合使用来在同一张画布上显示两幅图像
cv2.imshow("original image", img)
cv2.imshow("image show gray", img_gray)
cv2.waitKey(0)  # cv2.waitKey(0) 等待用户按下任何键。参数 0 表示无限期等待


# 调用包灰度化
plt.subplot(211)
img = cv2.imread("lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 如无这步，用plt.imshow读出颜色就会发生偏差
# img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# img_gray = rgb2gray(img)  # rgb2gray是scikit-image自带方法
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # cvtColor是opencv自带方法
plt.subplot(212)
plt.imshow(img_gray, cmap="gray")
print("---image gray----")
print(img_gray)
plt.show()
# 灰度化
# 方法一：将图像中的BGR坐标通过公式运算赋值给新的图像
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = cv2.imread("lenna.png")
# 获取图像的高度、宽度
height ,width = img.shape[:2]
# Numpy库创建一个与原图像相同大小的单通道图片  np.zero()函数用于创建一个具有指定形状的全零数组。[height,width]表示图像大小；image.dtype指定返回数组的数据类型，延用img的
img_gray = np.zeros([height,width], img.dtype)
for i in range(height):
    for j in range(width):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像    m[0],m[1],m[2]BGR模型

print (img_gray)
cv2.imshow("image show gray",img_gray)
cv2.waitKey()                                 # cv2.imshow()与waitKey()  合用显示图像


# # 方法二：使用opencv将彩色图像转换为灰度图像
# img = cv2.imread("lenna.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print (img_gray)
# cv2.imshow("image show gray",img_gray)
# cv2.waitKey()


# 二值化
# 方法一
rows , cols = img_gray.shape    #获趣灰度照片的行数和列数
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j]/255 <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

print(img_gray)
plt.imshow(img_gray,cmap='gray')          # plt.imshow(图像名，cmap='gray')显示灰度图片,与plt.show()合用显示照片
plt.show()

# 方法二
# img_binary = np.where(img_gray >= 0.5, 1, 0)
# print(img_binary)
# plt.imshow(img_gray,cmap='gray')
# plt.show()

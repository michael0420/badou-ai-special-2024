"""
@author: 207-xujinlan
实现彩色图像灰度化和二值化
"""

# 导包
from skimage.color import rgb2gray
import numpy as np
import cv2

# 方法一：使用python对图片进行灰度化
# 1.读入图片
img_path = 'lenna.png'
img = cv2.imread(img_path)

# 2.获取图片的高度（high)和宽度(wide)
h, w = img.shape[:2]

# 3.创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([h, w], img.dtype)

# 4.图片灰度化
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 获取彩色图片的BGR坐标
        img_gray[i, j] = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3  # 将彩色坐标值灰度化，赋值给单通道照片

print("image show original:%s" % img)  # 输出彩色图片数据
print("image show gray:%s" % img_gray)  # 输出灰度化图片数据
cv2.imshow("image show gray", img_gray)  # 展示图片
cv2.waitKey(0)

# 方法二：使用rgb2gray对图片进行灰度化
# 1.读入图片
img_path = 'lenna.png'
img = cv2.imread(img_path)
# 2.图片灰度化
img_gray = rgb2gray(img)  # 得到的是0-1之间的浮点数值
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #得到的是0-255之间的整数
cv2.imshow("image show gray", img_gray)  # 展示图片
cv2.waitKey(0)


# 图像二值化

# 方法一：使用for循环，这种方法效率低
# high,wide=img_gray.shape      #获取图像的高和宽
# img_binary=np.zeros([high,wide],img_gray.dtype)  # 创建一个空图片
# for i in range(high):
#     for j in range(wide):
#         if img_gray[i,j]>=0.5:    #阈值可以随意设置，浮点数和整数都可以，不一定要归一化
#             img_binary[i,j]=1
#         else:
#             img_binary[i,j]=0

# 方法二：使用np.where
img_binary = np.where(img_gray >= 0.5, 1, 0)
img_binary = (img_binary * 255).astype(np.uint8)  # imshow只能显示0-255的图像
cv2.imshow("image show binary", img_binary)  # 展示图片
cv2.waitKey(0)


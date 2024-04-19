import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# from PIL import Image

plt.subplot(221)                  # 多张图放在一张2*2画布上
imgPlt = plt.imread("lenna.png")  # 读取原始图像
plt.imshow(imgPlt)                # 把图像放在画布上
print("---image lenna----")
print(imgPlt)


"""
#  灰度化（计算式）
img = cv2.imread("lenna.png")                  #把读取图像
height,width = img.shape[:2]                   #获取图片的长和宽
img_gray = np.zeros([height,width],img.dtype)  #创建一张和当前图片大小一样的图片
for i in range(height):
    for j in range(width):
        m = img[i,j]                           #取出当前长和宽中的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  #将BGR坐标转化为gray坐标并赋值给新图像

print(m)
print(img_gray)
print("image show gray：%s"%img_gray)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')   # 把图像放在画布上
# cv2.imshow("image show gray",img_gray)
# cv2.waitKey(0)
"""

# 灰度化（调用函数式）
# img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(imgPlt, cv2.COLOR_BGR2GRAY) # 调用工具包函数直接把图像灰度化
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')    # 把图像放在画布上
print("---image gray func----")
print(img_gray)
# plt.show()                          #展示画布

"""
# 二值化（计算式）
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j] <= 0.5):
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1

print("---image_binary---")
print(img_gray)
print(img_gray.shape)

plt.subplot(325)
plt.imshow(img_gray, cmap='gray')
# plt.show()
"""

# 二值化（调用函数式）
img_binary = np.where(img_gray >= 0.5, 1, 0)    # 调用工具包函数直接把灰度化的图像二值化
print("---image_binary func---")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()                                    # 显示画布

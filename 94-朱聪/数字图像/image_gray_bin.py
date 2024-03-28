import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')  # 读取原始图片，注意cv2读取的是BGR，三通道  (500, 500, 3)
#
h, w = img.shape[:2]

# 灰度图转换   R0.3  G0.59  B0.11

gray_img = np.zeros((h, w), dtype=img.dtype)

for i in range(h):
    for j in range(w):
        m = img[i, j]
        gray_img[i, j] = int(0.11 * m[0] + 0.59 * m[1] + 0.3 * m[2])


# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 得到了灰度图，这个是没除255的，如果要在这个基础上转二值，还得先做归一化处理

# cv2.imshow("image show gray", gray_img)
# cv2.waitKey(0) # 不写会闪退或卡死

# gray_img = gray_img / 255 # 如果是手动实现的灰度图，得先除255，归一化，保证每个像素值在0-1之间

# gray_img = rgb2gray(img) # 这个方法转成灰度图，得到的像素是已经除了255的

# 手动实现二值图
# for i in range(h):
#     for j in range(w):
#         gray_img[i, j] = gray_img[i, j] > 0.5

# 直接调用得到二值图

# cv2.imshow("image show gray", gray_img)
# cv2.waitKey(0) # 不写会闪退或卡死


# ================================================================

# plt.subplot(221) # 生成画布,在2*2的画布上当前显示第一个位置的图像
# img = plt.imread('lenna.png')  # 原图
# plt.imshow(img)
#
#
# img_gray = rgb2gray(img)
# plt.subplot(222)
# plt.imshow(img_gray, cmap='gray') # 需要指定cmap的值
# plt.show()

gray_img = rgb2gray(img)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
bin_img = np.where(gray_img >= 0.5, 1, 0) # 了解np.where的使用
plt.subplot(223)
plt.imshow(bin_img, cmap='gray')
plt.show()
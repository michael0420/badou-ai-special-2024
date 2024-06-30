import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# 手写灰度化
img = cv2.imread("./Snipaste.png")
# h, w = img.shape[:2]               # 切片获取图片的行列数
# img_grey = np.zeros((h, w), img.dtype)         # 创建一个与原图大小一样的单通道图片
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]
#         img_grey[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)


# 调接口灰度化
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 一张画布显示一个图像
# cv2.imshow('image', img)
# cv2.imshow('grey_image', img_grey)
# cv2.waitKey(0)  # 等待用户按键输入，0表示无限等待
# cv2.destroyAllWindows()

# 一个画布显示多张图像：plt函数
plt.subplot(131)
img = plt.imread("Snipaste.png")
plt.imshow(img)

plt.subplot(132)
plt.imshow(img_grey, cmap='gray')

# 手动二值化
# h, w = img_grey.shape[:2]
# img_binary = np.zeros((h, w), img_grey.dtype)
# for r in range(h):
#     for c in range(w):
#         m1 = img_grey[r, c] / 255
#         if float(m1) <= 0.5:
#             img_binary[r, c] = 0
#         else:
#             img_binary[r, c] = 1

# 调用接口二值化
img_binary = np.where(img_grey/255 <= 0.5, 0, 1)

plt.subplot(133)
plt.imshow(img_binary, cmap='gray')
plt.show()













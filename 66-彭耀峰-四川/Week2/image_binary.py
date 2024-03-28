
"""
彩色图像二值化
"""

from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np

# 1.手动实现
def gray2binaryNew(img_gray):
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i, j] <= 0.5):
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 1
    return img_gray


img = plt.imread("lenna.png")
img_gray = rgb2gray(img)

img_binary1 = gray2binaryNew(img_gray)
plt.imshow(img_binary1,cmap='gray')
plt.show()

# 2.接口实现
# img_binary2 = np.where(img_gray >= 0.5, 1, 0)
# plt.imshow(img_binary2,cmap='gray')
# plt.show()
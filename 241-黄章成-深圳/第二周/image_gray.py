import cv2
import numpy
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

image = cv2.imread('lenna.png')

# 手写实现图片灰度化
h, w = image.shape[:2]
image_gray = numpy.zeros([h, w], image.dtype)
for i in range(h):
    for j in range(w):
        m = image[i, j]
        image_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

# print(m)
# print('origin image:', image)
# print('gray image: ', image_gray)
# print("image show gray: %s" % image_gray)
# cv2.imshow("gray_image", image_gray)
# cv2.waitKey(0)

# 可以存在本地，注意相对路径
# file_path = "processed_image.jpg"
# cv2.imwrite(file_path, image_gray)

# 使用opencv和skimage灰度化
image_gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 这里是用cv2实现图片灰度化
image_gray_sk = rgb2gray(image) # 这里是用skimage实现图片灰度化
# cv2.imshow("gray_image2", image_gray2)
# cv2.waitKey(0)

# file_path2 = "processed_image2.jpg"
# cv2.imwrite(file_path2, image_gray2)


# 手写实现图片二值化
row, col = image_gray.shape
image_binary = numpy.zeros([row, col], image_gray.dtype)
for i in range(row):
    for j in range(col):
        if image_gray[i, j]/255.0 <= 0.5:
            image_binary[i, j] = 0
        else:
            image_binary[i, j] = 255

# 使用opencv和skimage二值化
image_binary_cv = numpy.where(image_gray_cv/255.0 >= 0.5, 1, 0) # opencv的方式二值化，由于opencv返回的都是整数，所以得/255.0转化成浮点型
image_binary_sk = numpy.where(image_gray_sk >= 0.5, 1, 0) # skimage的方式二值化
# print('img_gray image: ', img_gray)
# print('binary image: ', image_binary)


# 将图片统一放在一个画布中方便比较
plt.subplot(331)
img = plt.imread('lenna.png')
plt.imshow(img)

plt.subplot(332)
plt.imshow(image_gray_cv, cmap='gray') # opencv 灰度化

plt.subplot(333)
plt.imshow(image_gray_sk, cmap='gray') # skimage 灰度化

plt.subplot(334)
plt.imshow(image_binary_cv, cmap='gray') # opencv 二值化

plt.subplot(335)
plt.imshow(image_binary_sk, cmap='gray') # skimage 二值化

plt.subplot(336)
plt.imshow(image_binary, cmap='gray') # 手动 二值化

plt.show()
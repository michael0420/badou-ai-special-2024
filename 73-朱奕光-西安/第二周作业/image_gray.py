import numpy
import cv2
import matplotlib.pyplot as plt

img_origin = cv2.imread('lenna.png')
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)  # 将读出的图转换成RGB形式
h, w = img.shape[:2]  # 读取原图长宽
# print(h,w)
img_gray = numpy.zeros([h, w], img.dtype)  # 创建空图

"""
灰度化
"""

for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
cv2.imshow("imge show grey", img_gray)
cv2.waitKey(0)

img_twokeys = numpy.zeros([h, w], img.dtype)  # 创建空图

"""
二值化
"""

for i in range(h):
    for j in range(w):
        img_twokeys[i, j] = img_gray[i, j]
        if img_twokeys[i, j] / 255 >= 0.5:
            img_twokeys[i, j] = 255
        else:
            img_twokeys[i, j] = 0
cv2.imshow("imge show twokeys", img_twokeys)
cv2.waitKey(0)

"""
plt视图输出
"""

plt.subplot(221)
img_origin_plt = plt.imread('lenna.png')
plt.imshow(img_origin_plt)

plt.subplot(222)
# img_gray_plt=plt.imread(img_gray)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
# img_twokeys_plt=plt.imread(img_twokeys)
plt.imshow(img_twokeys, cmap='gray')

plt.show()

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

#灰度化
x = cv.imread("lenna.png")
h, w = x.shape[:2]
cv.imshow("test", x)
#手动实现
# grayImg = np.zeros((h, w), x.dtype)
# for i in range(h):
#     for j in range(w):
#         m = x[i, j]
#         grayImg[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

#接口实现
grayImg = rgb2gray(x)
print(grayImg)

cv.imshow("test1", grayImg)
# cv.waitKey()

#二值化
#手动实现
# for i in range(h):
#     for j in range(w):
#         if grayImg[i, j] >= 0.5:
#             grayImg[i, j] = 1
#         else:
#             grayImg[i, j] = 0
#
# cv.imshow("test2", grayImg)
# cv.waitKey()
#调用接口实现
bvImg = np.where(grayImg >= 0.5, 1, 0)
bvImg = (bvImg * 255).astype(np.uint8)
# cv.imshow("test3", bvImg)
plt.imshow(bvImg, cmap="gray")
plt.show()
cv.waitKey()

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# 0：以灰度模式读取
# img2 = cv.imread('../lenna.png', 0)
img = cv.imread('../lenna.png', 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
dst = cv.equalizeHist(gray)
# print(dst)

# 直方图 calcHist
# - images: 是一个列表，包含要计算直方图的源图像。这些图像应该是相同大小和相同类型的。如果传入的图像是彩色的，那么每个通道都会被单独处理。
# - channels: 通道列表（对于灰度图，使用 [0]）
# - mask: 掩码图像（这里使用 None，计算整幅图像的直方图）
# - histSize: 是一个列表，指定了直方图的大小，即直方图条形的数量
# - ranges: 像素值范围（0到255），[0,256],结束值不包含在内
hist = cv.calcHist([dst], [0], None, [256], [0, 256])
print(hist)

plt.figure()
# ravel()方法是将多维数组展平为一维数组，这样我们就可以为整个图像的像素值绘制一个直方图。
# 表示直方图的箱子（bin）数量。
# 因为标准的8位灰度图像的像素值范围是0到255，所以这里设置箱子数量为256意味着每个箱子代表一个像素值，从而能够精确地展示每个像素值在图像中出现的次数。
plt.hist(dst.ravel(), 256)
plt.show()
# np.hstack 水平拼接
cv.imshow("", np.hstack([gray, dst]))
cv.waitKey(0)

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
# 把彩色图片灰度化，cv2.COLOR_BGR2GRAY参数就是把彩色图片灰度化
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("colorful",img)
# cv2.imshow("gray",gray)
# cv2.waitKey(0)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
cv2.imshow("gray",gray)
cv2.imshow("dst",dst)
cv2.waitKey(0)


# 直方图
# [dst]：这是输入图像的列表。在这个例子中，dst 是一个直方图均衡化后的灰度图像，
# 它被放在一个列表中作为输入。cv2.calcHist 可以处理多个图像，并将它们的直方图
# 合并起来，但在这个例子中我们只处理了一个图像。
# [0]：这是用于计算直方图的通道列表。因为 dst 是一个灰度图像，所以只有一个通道（即通道0）。
# 对于彩色图像（如BGR格式），你可能会有多个通道（0代表蓝色通道，1代表绿色通道，2代表红色通道），
# 并且可以选择计算哪个通道的直方图，或者计算所有通道的总直方图。
# None：这是掩模的列表。掩模用于指定图像中用于计算直方图的像素子集。如果传入 None，
# 则表示使用图像中的所有像素。如果你有一个特定的区域（例如，通过ROI定义的区域）并只
# 想计算这个区域的直方图，你可以创建一个与图像大小相同的二值掩模，其中你感兴趣的像素
# 值为255，其他像素值为0。
# [256]：这是直方图的bin数量。每个bin代表一个像素值范围，因此在这个例子中，直方图有256
# 个bin，每个bin对应一个可能的灰度级别（从0到255）。
# [0,256]：这是直方图的范围。第一个值（0）是直方图的最小值，第二个值（256）是直方图的最
# 大值。因为 dst 是一个8位灰度图像，所以其像素值范围是从0到255。设置为 [0,256] 是为了确
# 保直方图能够覆盖所有可能的像素值。尽管最大值设置为256超出了8位像素的实际最大值255，但这
# 通常不会导致问题，因为直方图函数知道如何处理这种情况。
# cv2.calcHist 函数返回的是一个数组，表示每个灰度级别的像素数量。因此，hist 是一个长度为
# 256的一维数组，其中 hist[i] 表示灰度级别为 i 的像素数量。这个数组可以用来绘制直方图，以
# 可视化图像中像素值的分布。
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()
cv2.waitKey(0)

# 显示直方图
# ravel()是一个数组方法，用于将多维数组展平（flatten）为一维数组
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.waitKey(0)

# np.hstack 是 NumPy（Numerical Python 的简称）库中的一个函数，用于水平堆叠
# 数组。它沿着第二个轴（列方向）将多个数组连接在一起。
# 具体来说，如果你有两个或多个数组，并且你想将它们沿着列方向拼接起来，你可以使
# 用 np.hstack。这与在 Excel 或其他表格处理软件中水平合并单元格的概念类似。
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(0)

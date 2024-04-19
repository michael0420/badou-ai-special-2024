import cv2

#
# 在OpenCV中，cv2.imread()函数用于读取图像文件。参数flag用于指定读取图像的方式：
#
# 如果flag为0（或cv2.IMREAD_GRAYSCALE），则以灰度模式读取图像，即将图像转换为单通道灰度图像。
# 如果flag为1（或cv2.IMREAD_COLOR），则以彩色模式读取图像，即保持原始图像的通道数（3通道）。
# 如果flag为-1（或cv2.IMREAD_UNCHANGED），则读取包含alpha通道的原始图像，如果图像包含alpha通道，则保持alpha通道。
img = cv2.imread('../Lenna.jpg', 0)
print(img.shape)

# CV_16S 16位有符号数  https://blog.csdn.net/dcrmg/article/details/52294259
# CV_16U 无符号数 还有带C1 C2 代表通道
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 转换成uint8
x = cv2.convertScaleAbs(x)
y = cv2.convertScaleAbs(y)

# alpha 和 beta 代表的是权重
res = cv2.addWeighted(x, 0.5, y, 0.5, 0)
print(res.shape)
cv2.imshow('x', x)
cv2.imshow('y', y)

cv2.imshow('res', res)
cv2.waitKey(0)
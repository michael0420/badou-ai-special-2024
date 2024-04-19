import cv2
import numpy as np

img = cv2.imread('lenna.png', 0) # 0表示以灰度模式读取一张图片
# 用cv2.CV_16S将原8位无符号数据，变为16位有符号数据类型，防止数据被截断
x = cv2.Sobel(img, cv2.CV_16S, 1, 0) # x方向求导
y = cv2.Sobel(img, cv2.CV_16S, 0, 1) # y方向求导

# 使用convertScaleAbs()函数将其转回原来的uint8形式。
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 使用cv2.addWeighted()函数将Sobel算子在x，y两个方向计算的数据结合起来

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)
cv2.waitKey(0)
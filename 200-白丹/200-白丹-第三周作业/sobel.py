import cv2

img = cv2.imread("lenna.png", 0)
# img：要计算梯度的图像。
# cv2.CV_16F：输出图像的深度，这里使用 16 位浮点数。
# 1 和 0：表示对图像在 x 方向和 y 方向计算梯度。
# ksize=3：Sobel 滤波器的大小，这里使用 3x3 的滤波器。

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 转化为uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
# 组合
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)
cv2.waitKey(0)
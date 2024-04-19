import cv2

# 读取图像，使用0表示读取灰度图
img = cv2.imread("../week2/lenna.png", 0)

# 使用Sobel算子计算图像的x方向和y方向的梯度
# 由于Sobel函数计算的结果可能为负数或超过255，因此使用16位有符号数据类型来避免截断，即cv2.CV_16S。
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 将Sobel算子计算得到的结果，用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 将x方向和y方向的梯度图像加权混合，得到最终的梯度图像，参数gamma含义为偏移量
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 显示各步骤的结果
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
cv2.imshow("Result", dst)

# 等待按键退出
cv2.waitKey(0)

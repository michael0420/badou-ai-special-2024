import cv2
import numpy as np

# 读取图片
img = cv2.imread("lenna.png", 0)

# 边缘检测
# cv2.Sobel() 是 OpenCV 中用于计算图像的一阶、二阶、三阶或混合图像导数的
# 函数，常用于边缘检测。这个函数基于 Sobel 运算符工作，该运算符是一个离散
# 微分运算符，用于计算图像灰度函数的近似梯度。
#
# 函数 cv2.Sobel() 的参数如下：
# img: 输入图像，必须是一个灰度图像。
# ddepth: 输出图像的深度。由于 Sobel 运算可能会产生负值，因此通常选择
# cv2.CV_16S、cv2.CV_32F 或 cv2.CV_64F 来确保负值能够被正确表示。在
# 后续的处理中，例如通过 cv2.convertScaleAbs() 转换为绝对值，通常会选
# 择 cv2.CV_16S。
# dx: x 方向上的导数阶数。
# dy: y 方向上的导数阶数。
# 以下代码 x = cv2.Sobel(img, cv2.CV_16S, 1, 0) 中：
# img 是输入图像。
# cv2.CV_16S 表示输出图像的深度是 16 位有符号整数，这样可以包含 Sobel 运算产生的负值。
# dx = 1 表示我们在 x 方向上计算一阶导数。
# dy = 0 表示我们不在 y 方向上计算导数。
# 因此，这段代码计算了图像 img 在 x 方向上的一阶导数，并将结果存储在 x 中。这通常用于
# 检测水平方向的边缘。由于输出图像是 16 位有符号整数，你可能需要调用 cv2.convertScaleAbs()
# 来将结果转换为 8 位无符号整数，并取绝对值，以便使用 cv2.imshow() 正确地显示结果。
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 因为导数可能存在负数，所以要取绝对值，否则有可能会出错
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 显示图像
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

# absX 和 absY 是两个输入图像或数组，它们的大小和类型应该相同。
# 0.5 是 absX 的权重。
# 0.5 是 absY 的权重。
# 0 是偏置值，这里表示不添加任何额外的偏置。
dst = cv2.addWeighted(absX, 0.5, absY, 0.5,0)
cv2.imshow("dst", dst)
cv2.waitKey(0)

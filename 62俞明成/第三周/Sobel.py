import cv2 as cv

img = cv.imread('../lenna.png', 0)
# cv.Sobel() 函数的参数如下：
# src：输入图像，通常应该是灰度图像。
# dst：输出图像，大小和类型与输入图像相同。
# depth：输出图像的深度，常用的有 cv.CV_8U（无符号8位）、cv.CV_16U（无符号16位）、cv.CV_16S（有符号16位
# cv.CV_32F（浮点数32位）和 cv.CV_64F（双精度浮点数64位）。通常，为了避免溢出，使用 cv.CV_64F 是个好选择。
# dx：表示沿 x 方向求导的阶数。
# dy：表示沿 y 方向求导的阶数。
# ksize：Sobel 核的大小，必须是 1, 3, 5, 或 7。
x = cv.Sobel(img, cv.CV_16S, 1, 0)
y = cv.Sobel(img, cv.CV_16S, 0, 1)

# convertScaleAbs()参数
# src：输入图像，通常是一个NumPy数组。
# alpha：缩放因子，控制输入数组元素的缩放比例。
# beta：添加到缩放后的值的偏移量。
# 输入图像的数据类型不是np.int8或np.uint8，这可能导致在取绝对值后像素值超出有效范围。
# 转为像素值的有效范围
absX = cv.convertScaleAbs(x)
absY = cv.convertScaleAbs(y)

# 将横纵方向计算的2个加起来
# addWeighted(src1, alpha, src2, beta, gamma, dst, dtype)
# src1：输入的第一张图片，即加权混合的第一张图片。
# alpha：第一张图片的权重。
# src2：输入的第二张图片，即加权混合的第二张图片。
# beta：第二张图片的权重。
# gamma：一个加到权重总和上的标量值。
# dst：输出的加权和图片。
# dtype：输出数组的深度，当参数为负数时，输出数组的深度与原图相同。
dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
cv.imshow("dst", dst)
cv.waitKey(0)
cv.destroyAllWindows()

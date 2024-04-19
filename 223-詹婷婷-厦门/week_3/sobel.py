import cv2
import numpy as np

#读取一幅图像 第一个参数是图像路径
#第二个参数代表读取方式， 1表示3通道彩色，0表示单通道灰度
img = cv2.imread("1947.jpg", 0)
cv2.imshow("img", img)


"""
Sobel()函数求完值之后，会有负值，以及大于255的值
而原图像是uint8，即8位无符号数(范围在[0.255])，所以Sobel建立的图像位数不够，会有截断
因此，需使用16位有符号的数据类型，即cv2.CV_16S

cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)

    1.src: 输入图像，即需要进行边缘检测的源图像。
    2.ddepth: 输出图像的深度（数据类型）。它通常设置为-1，表示输出图像的深度与输入图像相同。如果你希望输出的图像是有符号整数（Sobel算子通常会生成负数值），可以选择 cv2.CV_16S 或 cv2.CV_64F 作为输出深度。
    3.dx 和 dy: 分别表示Sobel算子的水平和垂直导数的阶数。通常，dx 设置为1，表示计算水平方向的导数；dy 设置为0，表示不计算垂直方向的导数。你也可以将它们互换以计算垂直导数。
    4.dst: 输出图像，用于存储Sobel操作的结果。这是可选参数，可以省略。
    5.ksize: Sobel核的大小，通常是3，表示3x3的核。这个参数决定了Sobel算子的灵敏度。
    6.scale: 可选参数，用于缩放导数的值。通常设置为1。通过调整这个参数，你可以控制导数值的幅度。
    7.delta: 可选参数，添加到输出图像上的可选值。通常设置为0。
    8.borderType: 可选参数，边界处理的类型，可以是以下之一：
        cv2.BORDER_DEFAULT: 默认边界处理方式，通常使用0值填充。
        cv2.BORDER_CONSTANT: 使用用户指定的常数值进行填充。
        cv2.BORDER_REPLICATE: 使用最边缘像素进行填充。
        cv2.BORDER_REFLECT: 使用图像的反射边界像素进行填充。
"""

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

"""
经过cv2.Sobel(img, cv2.CV_16S, 1, 0)处理后，
使用cv2.convertScaleAbs(x)将其转回原来的uint8形式
"""
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

"""
通过cv2.addWeighted(absX, 0.5, absY, 0.5, 0)，将两个方向的计算组合起来
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
"""
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("dst", dst)

cv2.waitKey()

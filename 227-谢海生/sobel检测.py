import cv2
import numpy as np

img = cv2.imread("C:/Users/86188/Pictures/lenna.png",0)

'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
'''

#cv2.Sobel() 函数的基本语法：

#cv2.Sobel(src, ddepth, dx, dy[, ksize[, scale[, delta[, borderType]]]])

'''
src 是输入图像。
ddepth 是输出图像的深度。在这个例子中，它被设置为 cv2.CV_16S，表示输出图像的深度为 16 位整数。
dx 和 dy 分别表示 X 和 Y 方向上的梯度。在这个例子中，它们分别被设置为 1 和 0，表示计算 X 方向的梯度。
ksize 是 Sobel 滤波器的大小。在这个例子中，它被省略，表示使用默认的滤波器大小。
scale 是滤波器的缩放因子。在这个例子中，它被省略，表示使用默认的缩放因子。
delta 是输出图像的偏移量。在这个例子中，它被省略，表示使用默认的偏移量。
borderType 是处理图像边界时使用的插值方法。在这个例子中，它被省略，表示使用默认的插值方法。
 cv2.Sobel() 函数计算图像的梯度，并返回 X 和 Y 方向的梯度图像。
 在这个例子中，X 方向的梯度图像被存储在 x 变量中，Y 方向的梯度图像被存储在 y 变量中
'''
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

'''
用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''
#计算梯度
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。

src1和src2是输入图像。
alpha和beta是权重，用于分别缩放src1和src2。在这个例子中，它们都被设置为0.5，表示两个图像的权重相等。
gamma是偏移量，用于调整混合后的图像的亮度。在这个例子中，它被设置为0，表示没有偏移量。
dst是输出图像。在这个例子中，它被省略，表示混合后的图像将自动创建并存储在dst变量中。
dtype是输出图像的数据类型。在这个例子中，它被省略，表示输出图像的数据类型将与输入图像相同。
 cv2.addWeighted()函数将两个图像加权混合在一起，并返回混合后的图像。
 在这个例子中，它将absX和absY图像加权混合在一起，并将结果存储在dst变量中。 
 最后，使用cv2.imshow()函数显示absX、absY和混合后的图像dst。

'''

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
#OpenCV程序的末尾调用cv2.destroyAllWindows()函数，以确保在程序退出之前关闭所有窗口
cv2.destroyAllWindows()
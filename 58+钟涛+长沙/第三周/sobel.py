import cv2
import numpy as np

#读取灰度图片
img = cv2.imread("lenna.png", 0)

'''
cv2.CV_16S是一个数据类型常量，表示16位有符号整数（Short）。
这种数据类型在图像处理中经常用于存储可能包含负数的中间结果。
例如，在进行图像滤波操作如Sobel边缘检测时，结果可能包含负值，这时使用cv2.CV_16S可以确保这些负值不会丢失。
'''
#x方向梯度
sobelx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
#y方向梯度
sobely = cv2.Sobel(img,cv2.CV_16S, 0, 1)

#转换 unit8
'''
cv2.convertScaleAbs是OpenCV中用于转换图像深度并取绝对值的函数。
这个函数主要用于在进行图像处理操作后（如梯度计算）将结果转换为对于显示和存储更友好的格式。
它的常见用途包括将图像从带有负值的数据类型（如cv2.CV_16S）转换为无符号的8位类型（cv2.CV_8U），同时保留信息的关键部分。
'''
absx = cv2.convertScaleAbs(sobelx)
absy = cv2.convertScaleAbs(sobely)

# 图片叠加
'''
在OpenCV中，cv2.addWeighted函数用于按照指定的权重将两个图像相加，通常用于图像融合或者合并不同的图像处理效果。
这个函数也可以用来结合两个方向上的梯度图像，如Sobel边缘检测的结果。

'''
dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

# cv2.imshow("img", img)
# cv2.imshow("x", absx)
# cv2.imshow("y", absy)
# cv2.imshow("dst", dst)
#显示图片
cv2.imshow("sobel", np.hstack([img, absx, absy, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()



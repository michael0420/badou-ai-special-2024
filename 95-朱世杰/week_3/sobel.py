"""
Sobel边缘检测
@AuThor: zsj
"""
import cv2
from matplotlib import pyplot as plt

# 读取灰度图像
img = cv2.imread("../lenna.png", 0)

'''
img: 输入图像，它应该是一个灰度图像。如果输入是彩色图像，需要先将其转换为灰度图像。
cv2.CV_16S: 输出图像的深度。CV_16S 表示输出图像的数据类型是有符号的16位整数。选择这个深度是因为Sobel函数可能会产生负值，而8位图像无法表示负值。使用16位深度可以保留负值，稍后可以通过取绝对值或转换回8位图像来显示。
1: xorder 参数，表示沿x方向的差分阶数。在这里，它是1，意味着我们计算x方向的一阶导数。
0: yorder 参数，表示沿y方向的差分阶数。在这里，它是0，意味着我们不在y方向计算导数。
'''
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

'''
计算绝对值，转换类型并归一化到0-255的范围内，用convertScaleAbs()函数将其转回原来的uint8形式
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

plt.figure()
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.subplot(222)
plt.imshow(absX, cmap='gray')
plt.subplot(223)
plt.imshow(absY, cmap='gray')
plt.subplot(224)
plt.imshow(dst, cmap='gray')
plt.show()
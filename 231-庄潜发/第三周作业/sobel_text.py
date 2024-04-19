"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/9 23:18
"""
import cv2

img = cv2.imread("lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""
sobel函数求完导数后会有负值,还会有大于255的值
而原图像是uint8(0-255),所以会有截断
因此我们要使用16位有符号的数据类型, cv2.CV_16S
"""
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 1, 0 x方向
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 0, 1 y方向
"""
处理后,需要用convertScaleAbs()函数将其转换为uint8类型的图片
cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
alpha位伸缩系数, beta是加到结果上一个值, 结果返回uint8图像
否则无法显示图像
"""
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
"""
由于sobel两个方向计算,最后还需要cv2.addWeighted()将其组合起来
cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
其中alpha位第一幅图片中的元素权重, beta是第二个的权重
gamma是加到最后结果上的一个值
"""
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow('absX', absX)
cv2.imshow('absY', absY)
cv2.imshow('result', dst)

cv2.waitKey()
cv2.destroyAllWindows()

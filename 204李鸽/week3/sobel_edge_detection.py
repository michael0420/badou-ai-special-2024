import cv2
img = cv2.imread('girl.png')
'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号整数的数据类型，即cv2.CV_16S。
'''
# 卷积公式里有积分，导数的逆运算
# x, y方向边缘检测
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# 如果不需要可视化到这里就结束了
'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''
X = cv2.convertScaleAbs(x)
Y = cv2.convertScaleAbs(y)
'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''
XY = cv2.addWeighted(X, 0.5, Y, 0.5, 0)
cv2.imshow('X', X)
cv2.imshow('Y', Y)
cv2.imshow('XY', XY)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
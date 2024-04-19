import cv2

img1 = cv2.imread("yangmi.jpg", 0) #读图
img = cv2.resize(img1, (540, 960)) #缩放

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

"""
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。

dst = cv2.Sobel( src, ddepth, dx, dy[,ksize[, scale[, delta[, borderType]]]] )
式中：
 dst 代表目标图像。
 src 代表原始图像。
 ddepth 代表输出图像的深度。
 dx 代表 x 方向上的求导阶数。
 dy 代表 y 方向上的求导阶数。
 ksize 代表 Sobel 核的大小。该值为-1 时，则会使用 Scharr 算子进行运算。
 scale 代表计算导数值时所采用的缩放因子，默认情况下该值是 1，是没有缩放的。
 delta 代表加在目标图像 dst 上的值，该值是可选的，默认为 0。
 borderType 代表边界样式。
"""

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

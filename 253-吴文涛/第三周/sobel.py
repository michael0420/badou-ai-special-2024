import cv2
img = cv2.imread("image/lenna.png", 0)
#sobel求导后有负数，uint8没有符号，sobel函数图像位数不够会被截断，因此要转换为16位cv2.CV_16S
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
#转换为uint8 否则无法显示图像
#dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
#由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
#dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
#其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)
cv2.waitKey(0)

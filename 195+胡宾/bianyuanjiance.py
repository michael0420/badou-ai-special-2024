import cv2
# 读取图片，灰度化
picture = cv2.imread("lenna.png", 0)
# cv2.imshow("src", picture)
# cv2.waitKey(0)
# 横向边缘检查，原图是8位无符号数(范围在[0,255])，而sobel使用的16位无符号数，所以使用cv2.CV_16S
transverse = cv2.Sobel(picture, cv2.CV_16S, 1, 0)
# 纵向边缘检查
portrait = cv2.Sobel(picture, cv2.CV_16S, 0, 1)

# 展示图像，需使用convertScaleAbs()函数将其转回原来的uint8形式
abs_x = cv2.convertScaleAbs(transverse)
abs_y = cv2.convertScaleAbs(portrait)


# 由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
# 。其函数原型为：
# dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
# 其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
# gamma是加到最后结果上的一个值。

sobel_dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
cv2.imshow("abs_x", abs_x)
cv2.imshow("abs_y", abs_y)
cv2.imshow("result", sobel_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
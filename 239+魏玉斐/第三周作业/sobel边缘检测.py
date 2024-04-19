import cv2
import matplotlib.pyplot as plt

# 读取图像,0表示以灰度模式读取
img = cv2.imread('lenna.png', 0)
'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
'''
# 使用Sobel算子进行水平边缘检测
x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0)
# 使用Sobel算子进行垂直边缘检测
y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1)
'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
# 合并两个边缘图像
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(222)
plt.imshow(absX, cmap='gray')
plt.title('Sobel Edge Image (X)')
plt.subplot(223)
plt.imshow(absY, cmap='gray')
plt.title('Sobel Edge Image (Y)')
plt.subplot(224)
plt.imshow(dst, cmap='gray')
plt.title('Total Edge Image')
plt.show()

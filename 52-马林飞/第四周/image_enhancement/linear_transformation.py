import cv2
import numpy as np

src_image = cv2.imread('lenna.png')

# y=a*x+b  a影响图像的对比度，b影响图像的亮度

alpha = 1.5
beta = 1

enhanced_image = cv2.convertScaleAbs(src_image, alpha, beta)
cv2.imshow('src-dest', np.hstack([src_image, enhanced_image]))
cv2.waitKey(0)
cv2.destroyAllWindows()

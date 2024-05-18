import cv2
import numpy as np

img = cv2.imread("1.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
调用cv2.Canny()的接口：
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
'''

cv2.imshow("canny", cv2.Canny(gray, 200, 300))
cv2.waitKey(0)
cv2.destroyAllWindows()

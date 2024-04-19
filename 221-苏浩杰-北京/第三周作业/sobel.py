"""
图像边缘检测，包括横向和纵向
"""

import  cv2

"""
flag > 0 返回一个3通道的彩色图像
flag = 0 返回灰色图像
flag < 0 返回包含Alpha通道的加载图像
"""
imread = cv2.imread("../lenna.png", 1)

x=cv2.Sobel(imread, cv2.CV_16S, 1, 0)
y=cv2.Sobel(imread, cv2.CV_16S, 0,1)
# xy=cv2.Sobel(imread,cv2.CV_16S,1,1)


abs_x=cv2.convertScaleAbs(x)
abs_y=cv2.convertScaleAbs(y)
# abs_xy=cv2.convertScaleAbs(xy)

dst=cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

cv2.imshow("x",abs_x)
cv2.imshow("y",abs_y)
# cv2.imshow("xy",abs_xy)
cv2.imshow("dst",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



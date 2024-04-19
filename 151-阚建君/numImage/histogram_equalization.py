import numpy as np
import cv2

'''
实现直方图均衡化
'''

# 灰度直方图均衡化
# img = cv2.imread("lenna.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# equal_img = cv2.equalizeHist(img_gray)
# result = np.hstack((img_gray, equal_img))
# cv2.imshow("gray equalization", result)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 彩色图像均衡化
# img = cv2.imread("lenna.png")
# b, g, r = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# result = cv2.merge((bH,gH,rH))
# cv2.imshow("original", img)
# cv2.imshow("color equalization", result)
# cv2.waitKey()
# cv2.destroyAllWindows()



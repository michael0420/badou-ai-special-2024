import numpy as np
import cv2


def CannyThreshold(lower_threshold):
    # canny边缘检测
    canny_img = cv2.Canny(gray, lower_threshold, lower_threshold * 3, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=canny_img)
    cv2.imshow('canny result', dst)


kernel_size = 3
# 这里是调节杆的能调节的范围，并不是双阈值算法中的高低阈值，所以max_threshold不能置为0，否则调节不了
min_threshold = 0
max_threshold = 100
# 以灰度化的形式读图
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 这里的窗口的名字必须与下面的第二个参数的名字一致，否则会报错
cv2.namedWindow('canny result')

# cv2.createTrackbar('canney_trackbar', 'canny result', lower_threshold, high_threshold, CannyThreshold)
cv2.createTrackbar('Min threshold', 'canny result', min_threshold, max_threshold, CannyThreshold)

CannyThreshold(0)
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()

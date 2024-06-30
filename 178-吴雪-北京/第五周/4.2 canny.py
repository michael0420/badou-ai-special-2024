"""
Canny invocation interface
"""
import cv2

img = cv2.imread('E:/Desktop/jianli/lenna.png', 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('canny', cv2.Canny(gray, 200, 300))
cv2.waitKey(0)
cv2.destroyAllWindows()

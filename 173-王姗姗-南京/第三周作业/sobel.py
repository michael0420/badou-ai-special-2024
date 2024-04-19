import cv2

# sobel边缘检测

img = cv2.imread('lenna.png', 0)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# x轴边缘检测显示
cv2.imshow("absX", absX)
# y轴边缘检测显示
cv2.imshow("absY", absY)

# 全图边缘检测显示
cv2.imshow("result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

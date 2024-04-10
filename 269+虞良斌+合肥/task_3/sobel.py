import cv2

'''sobel边缘检测'''

img = cv2.imread("../lenna.png", 0)

# sobel函数
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

sob_X = cv2.convertScaleAbs(x)
sob_Y = cv2.convertScaleAbs(y)

# x, y方向叠加
dst = cv2.addWeighted(sob_X, 0.5, sob_Y, 0.5, 0)

cv2.imshow("absX", sob_X)
cv2.imshow("absY", sob_Y)
cv2.imshow("Result", dst)

cv2.waitKey(0)


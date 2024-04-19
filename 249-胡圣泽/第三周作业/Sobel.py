import cv2


img = cv2.imread('lenna.png')

#灰度化后再卷积
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#横向/纵向边缘检测，利用边缘检测接口Sobel
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

#CV_16S 16位有符号数据类型，通过convertScaleAbs()转换回uint8形式，才能正常显示
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

#将两个方向的边缘检测结果组合
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()



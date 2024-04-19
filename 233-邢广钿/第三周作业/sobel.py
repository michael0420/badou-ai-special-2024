

import cv2

img = cv2.imread("lenna.png", 0)

# 分别得到z、y方向的sobel  cv2.CV_16S：输出有符号的16位的图像(原图为uint8[0,255],sobel的计算可能会超出原图)
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# 计算梯度绝对值并转换为原来的unit8图像
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 按x、y分别为0.5的权重进行合并
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

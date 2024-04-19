import cv2

img = cv2.imread("lenna.png", 0) #flag等于0读入灰度图

#分别在x和y方向建立Sobel算子 避免产生截断 使用32位有符号数据类型
x = cv2.Sobel(img, cv2.CV_32F, 1, 0) #x方向
y = cv2.Sobel(img, cv2.CV_32F, 0, 1) #y方向

#Sobel处理完成 转回utf-8显示图像
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

#将x和y方向分别Sobel的图像各0.5权重相加
dst = cv2.addWeighted(x, 0.5, y, 0.5, 0); # x 0.5权重 y 0.5权重 不加额外像素值

#显示图像
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
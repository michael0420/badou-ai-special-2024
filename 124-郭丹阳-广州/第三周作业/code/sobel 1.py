import cv2

# 加载图像
img = cv2.imread('D:/AI--pan/code/lenna.png', cv2.IMREAD_GRAYSCALE)

# Sobel算子进行边缘检测
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# 将结果转换为绝对值
abs_sobel_x = cv2.convertScaleAbs(sobel_x)
abs_sobel_y = cv2.convertScaleAbs(sobel_y)

# 将水平和垂直方向的边缘检测结果组合起来
sobel_edges = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# 显示水平和垂直方向的边缘检测结果以及组合后的结果
cv2.imshow("Sobel X", abs_sobel_x)
cv2.imshow("Sobel Y", abs_sobel_y)
cv2.imshow("Sobel Edges", sobel_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

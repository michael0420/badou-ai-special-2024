"""
Sobel边缘检测    Sobel edge detection
"""
import cv2
img = cv2.imread("C:/Users/16040/Desktop/CV_/badou-ai-special-2024/335-鲁杨森-上海/week3/lenna.png", 0)  # 获取灰色图像
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 对x求导  使用Sobel算子进行水平边缘检测
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 对y求导  使用Sobel算子进行垂直边缘检测
# 将Sobel水平和垂直边缘检测结果的负数值转换为正数。结果返回uint8类型的图片
x = cv2.convertScaleAbs(x)
y = cv2.convertScaleAbs(y)
# 使用addWeighted函数将水平和垂直边缘检测结果叠加，权重都为0.5，创建合并的边缘检测图像
sobel = cv2.addWeighted(x, 0.5, y, 0.5, 0)
cv2.imshow("Horizontal edge detection", x)
cv2.imshow("Vertical edge detection", y)
cv2.imshow("Sobel edge detection", sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()

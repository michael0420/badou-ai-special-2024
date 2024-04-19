"""
彩色图像直方图均衡化 histogram equalization (color)
"""
import cv2
img = cv2.imread("C:/Users/16040/Desktop/CV_/badou-ai-special-2024/335-鲁杨森-上海/week3/lenna.png", 1)
cv2.imshow("src", img)
# 彩色图像均衡化，需要分解通道，对每一个通道均衡化
(b, g, r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bh, gh, rh))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
直方图均衡化 histogram equalization (gray)
作用是图像增强
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("C:/Users/16040/Desktop/CV_/badou-ai-special-2024/335-鲁杨森-上海/week3/lenna.png", 1)  # 读取彩色图像为1
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.plot(hist, color='r')
plt.figure()  # 创建画布
plt.hist(dst.ravel(), 256)  # hist()函数根据数据源和像素级绘制直方图
plt.show()
# np.hstack()函数，用于将两个或更多的数组沿着水平轴连接起来
cv2.imshow("histogram equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()

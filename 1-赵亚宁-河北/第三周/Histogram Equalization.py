import cv2
import numpy as np
from matplotlib import pyplot as plt
# 读取图像，转换为灰度图
image = cv2.imread('lenna.png', 0)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 应用直方图均衡化
equalized_image = cv2.equalizeHist(image)


# 直方图
hist = cv2.calcHist([equalized_image],[0],None,[256],[0,256])

plt.figure()
plt.hist(equalized_image.ravel(), 256)
plt.show()

# 显示原图和均衡化后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 如果需要，保存均衡化后的图像
cv2.imwrite('equalized_image.jpg', equalized_image)
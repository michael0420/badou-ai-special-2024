import cv2  
import numpy as np  
from matplotlib import pyplot as plt  
  
# 获取灰度图像  
img = cv2.imread("scaled_binary_image.jpg", cv2.IMREAD_COLOR)  # 确保读取为彩色图像  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像  
  
# 灰度图像直方图均衡化  
dst = cv2.equalizeHist(gray)  
  
# 计算并显示原始灰度图像的直方图  
plt.figure(figsize=(12, 6))  # 设置图形大小  
  
# 创建第一个子图显示原始灰度图像的直方图  
plt.subplot(1, 2, 1)  # 1行2列的子图，当前是第1个  
plt.hist(gray.ravel(), 256, [0, 256])  
plt.title('Original Histogram')  
plt.xlabel('Pixel Intensity')  
plt.ylabel('Frequency')  
  
# 创建第二个子图显示均衡化后图像的直方图  
plt.subplot(1, 2, 2)  # 1行2列的子图，当前是第2个  
plt.hist(dst.ravel(), 256, [0, 256])  
plt.title('Equalized Histogram')  
plt.xlabel('Pixel Intensity')  
plt.ylabel('Frequency')  
  
# 显示图形  
plt.tight_layout()  # 调整子图之间的间距  
plt.show()  
  
# 显示原始灰度图像和均衡化后的图像  
cv2.imshow("Original vs Equalized", np.hstack([gray, dst]))  
cv2.waitKey(0)  
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


'''
# 彩色图像直方图均衡化
img = cv2.imread("scaled_binary_image.jpg", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''
import cv2
import matplotlib.pyplot as plt

# 读取图像
# img=cv2.imread('lenna.png',0)#0灰度，默认为1，彩色图
# opencv对于读入的矩阵是BGR
# 设置全局中文字体
# plt.rcParams['font.sans-serif'] = ['SimSun']
img = cv2.imread('lenna.png')
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
# 灰度图像进行直方图均衡化
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_grey_equalized = cv2.equalizeHist(img_grey)
cv2.imshow("original", img)
cv2.imshow("dst_rgb", result)
cv2.imshow("dst_grey", img_grey_equalized)
cv2.waitKey(0)

# 画出原图的直方图
# 计算彩色图的直方图

plt.subplot(221)
plt.hist(img_grey.ravel(), 256, (0, 256))
plt.title('Histogram for original image')
# 画出均衡化后的直方图
plt.subplot(222)
plt.hist(result.ravel(), 256, (0, 256))
plt.title('Histogram for result image')
plt.subplot(223)
plt.hist(img_grey_equalized.ravel(), 256, (0, 256))
plt.title('Histogram for img_grey image')
plt.show()

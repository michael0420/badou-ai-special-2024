import cv2
import matplotlib.pyplot as plt

img = cv2.imread('yeah.png')
# 对彩色图像三个通道分别做均衡化
b, g, r = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

# 计算彩色图像直方图的方法
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

# 合并三个通道的直方图
hist = cv2.merge((hist_b, hist_g, hist_r))

# 绘制直方图
plt.figure()
plt.plot(hist_r, color='red', label='R', linewidth=2)
plt.plot(hist_g, color='green', label='G', linewidth=2)
plt.plot(hist_b, color='blue', label='B', linewidth=2)
plt.title('RGB Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# # 将3个通道合并起来
# img_h = cv2.merge((bh, gh, rh))
# cv2.imshow('src', img)
# cv2.imshow('dst', img_h)
# cv2.waitKey(0)

# 灰度图均衡化
# 获取灰度图
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# # 灰度图直方图均衡化
# gray_h = cv2.equalizeHist(img_gray)
# gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
# # 创建一个新的图形窗口
# plt.figure()
# # # 绘制灰度图像的直方图
# # plt.plot(gray_hist, color='black', linewidth=2)
# # # 设置图形标题和坐标标签
# # plt.title('Grayscale Image Histogram')
# # plt.xlabel('Pixel value')
# # plt.ylabel('Frequency')
# # # 显示图形
# plt.show()

# # 柱状图显示
# plt.figure()
# plt.hist(gray_hist.ravel(), 256)
# plt.show()
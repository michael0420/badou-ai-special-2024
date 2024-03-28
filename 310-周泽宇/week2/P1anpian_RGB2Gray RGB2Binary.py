import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用cv2库读取图片
img_BGR = cv2.imread("../lenna.png") # cv2读入的图片为BGR格式
cv2.imshow('CV2_Image', img_BGR) # 显示图片 cv2.imshow也默认按照BGR格式输出图片
cv2.waitKey() # 显示后等待键盘事件再消失

# 转换为灰度图像
h, w = img_BGR.shape[:2] # img.shape()返回一个三元组分别表示 高度 宽度 颜色通道数
# 取前两个维度分别为高度和宽度
img_gray = np.zeros([h,w], img_BGR.dtype)

for i in range(h):
    for j in range(w):
        m = img_BGR[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray, cmap='gray') #显示灰度图像
plt.show()
print(img_gray)

# 转换为二值图像
img_binary = np.where(img_gray >= 128, 1, 0)
plt.imshow(img_binary, cmap='gray') #显示灰度图像
plt.show()
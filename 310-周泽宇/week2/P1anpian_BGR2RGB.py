import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用cv2库读取图片
img_BGR = cv2.imread("../lenna.png") # cv2读入的图片为BGR格式
cv2.imshow('CV2_Image', img_BGR) # 显示图片 cv2.imshow也默认按照BGR格式输出图片
cv2.waitKey() # 显示后等待键盘事件再消失

'''Part1 BGR2RGB'''
# 转换为RGB格式①
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
# 转换为RGB格式②
img_RGB_2 = img_BGR[:, :, ::-1]
'''
第一个冒号:表示取所有行；
第二个冒号:表示取所有列；
::-1表示在第三个维度（颜色通道）上，将元素的顺序反转。
'''

# 显示 RGB 图像
plt.figure() # 创建一个新的图形窗口

plt.subplot(1,2,1) # 在第一个子图中显示第一个图像
plt.imshow(img_RGB) # imshow将矩阵数据渲染成图片

plt.subplot(1,2,2) # 在第二个子图中显示第二个图像
plt.imshow(img_RGB_2)

plt.show() # 显示图形
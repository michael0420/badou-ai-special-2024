import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
# img是三维numpy数组，形状为(height, width, 3)
img = cv2.imread('lenna.png')

# 将img三维数组重塑为二维数组img_one
img_one = img.reshape((-1,3))
'''
reshape((-1, 3))具体含义:
    -1 表示自动计算这一维度的大小，以确保重塑后的数组总元素数量与原始数组相同
    3 表示每一行有三个值，分别对应一个像素的 BGR 颜色通道
这样 img_one 的形状将变为 (height * width, 3)，原来的三维图像数据被展平为一个二维数组，其中每一行表示一个像素的颜色值
目的：每个像素点的 BGR 颜色值能够作为一个独立的数据点进行 k-means 聚类
'''

img_one = np.float32(img_one)  # kmeans算法所需要的输入格式

# 停止条件(type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# Kmeans聚类
# 将img_one中每一行的BGR值作为一个数据点进行聚类，分为4个簇
compactness, labels, centers = cv2.kmeans(img_one, 4, None, criteria, 10, flags)
'''
输出参数
1.compactness: 距离值（紧密度）即累计平方误差，返回每个点到相应中心距离的平方和
2.labels: 每个像素点的聚类标签
3.centers: 每个分类的中心点数据
'''

# 重构图像
centers = np.uint8(centers)  # centers为K个聚类中心的颜色值，需转换为图像显示所需要的uint8格式
res = centers[labels.flatten()]  # 将数组labels展平成一维，并用聚类标签labels将每个像素点的颜色替换为其对应聚类中心的颜色
dst = res.reshape((img.shape))  # 将一维数组res重塑为与原始图像相同的形状

# 转换为RGB显示
# OpenCV中默认颜色顺序是BGR，而Matplotlib中颜色顺序是RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# 正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
plt.subplot(1, 2, 1)  # 121:一行两列第一个位置
plt.imshow(img)
plt.title('原始图像')
plt.xticks([]), plt.yticks([])  # 隐藏x轴和y轴的刻度坐标

plt.subplot(1, 2, 2)
plt.imshow(dst)
plt.title('聚类图像 K=4')
plt.xticks([]), plt.yticks([])

plt.show()

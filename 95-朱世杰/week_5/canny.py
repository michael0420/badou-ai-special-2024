"""
实现canny算法
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

# 1.对图像进行灰度化
img = cv2.imread("../lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2.高斯平滑
sigma = 0.5
dim = 5  # 推荐高斯核尺寸
Gaussian_filter = np.zeros([dim, dim])  # 高斯核
# 高斯核计算公式中的两个参数
n1 = 1 / (2 * math.pi * sigma ** 2)
n2 = -1 / (2 * sigma ** 2)
mid = dim // 2
# 计算高斯核
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i, j] = n1 * math.exp(n2 * ((i - mid) ** 2 + (j - mid) ** 2))  # 高斯核最大值在中心
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
# 进行高斯平滑
dx, dy = img.shape
img_new = np.zeros(img.shape)  # 平滑后图像
pad = dim // 2  # 计算步长为1时，padding的值
img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'constant')  # 边缘填补
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
plt.figure(1)
plt.subplot(221)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

# 3.使用sobel卷积核求梯度
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros([dx, dy])
img_tidu_y = np.zeros([dx, dy])
img_tidu = np.zeros([dx, dy])
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
for i in range(dx):
    for j in range(dy):
        img_block = img_pad[i:i + 3, j:j + 3]
        img_tidu_x[i, j] = np.sum(img_block * sobel_kernel_x)  # x方向
        img_tidu_y[i, j] = np.sum(img_block * sobel_kernel_y)  # y方向
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
img_tidu_x[img_tidu_x == 0] = 0.00000001
angle = img_tidu_y / img_tidu_x
plt.subplot(222)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4.非极大值抑制
img_yizhi = np.zeros([dx, dy])
# 边界上没有足够的邻域，不用计算边界
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # 是否为极大值的标签
        temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的3*3矩阵
        # 根据tan值判断梯度方向与3*3的交点位置
        if angle[i, j] <= -1:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:  # 只保留极大值的点
            img_yizhi[i, j] = img_tidu[i, j]
plt.subplot(223)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 5.双阈值检测
# 设置双阈值
lower_boundary = 100
high_boundary = 200
zhan = []
for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:  # 取>=高阈值的点
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  # 舍
            img_yizhi[i, j] = 0

while len(zhan) > 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    # 把所有强边缘点周边的弱边缘点 循环编辑，并标为强边缘点
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if (value < high_boundary) and (value > lower_boundary):
                img_yizhi[temp_1 + (i - 1), temp_2 + (j - 1)] = 255
                zhan.append([temp_1 + (i - 1), temp_2 + (j - 1)])

# 其他的弱边缘点直接舍弃
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

# 绘图
plt.subplot(224)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值
plt.show()
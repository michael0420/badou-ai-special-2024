import numpy as np
import matplotlib.pyplot as plt
import math

"""
Canny 相比于Sobel和Prewitt算子而言 更容易找到最优边缘
1. 对图像进行灰度化（只是为了降低计算复杂度 非必要 特殊的如红绿灯检测可以保留彩色图）
2. 对图像进行高斯滤波（目的和上一步一样 同样非必须）根据高斯函数对灰度值加权
3. 检测图像边缘（Sobel或Prewitt等）
4. 对梯度非极大值抑制
5. 双阈值检测和连接边缘

非极大值抑制：将当前像素梯度与沿正负梯度的两个像素比较 如果它是最大值 则保留 否则抑制置0
双阈值检测：如果边缘梯度高于高阈值，标为强边缘
	     如果边缘梯度小于高阈值且大于低阈值，标为弱边缘
	     如果边缘梯度小于低阈值，抑制
	     为了防止噪声这样的独立边缘点形成的假边缘， 应该抑制所以通过检测周围相邻八个点
	     是否全是弱边缘 全是则抑制
"""

#1.读取lenna.png 并对图像进行灰度化
pic_path = "lenna.png"
img = plt.imread(pic_path)

#读取的png文件在存储格式为0-1之间的浮点数 转化为0-255范围内 进行均值灰度化 axis = -1代表对最后一维操作
if(pic_path[-4:] == ".png"):
    img = img * 255
img = img.mean(axis=-1)

#2. 高斯滤波
#设置高斯滤波的相关参数
sigma = 0.5 # 标准差
dim = 5 #卷积核大小
Gaussian_filter = np.zeros([dim,dim]) #生成5*5全零卷积核数组
#带入公式 计算高斯核
n1 = 1/(2*math.pi*sigma**2)
n2 = -1/(2*sigma**2)
#生成一个序列 减去半径代表领域内各点与中心点坐标差值
tmp = [i-dim//2 for i in range(dim)]
#遍历卷积核中的每一个点 并做高斯加和
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
#归一化
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
dx, dy = img.shape
new_img = np.zeros(img.shape) #生成符合图像大小的全零矩阵 存储浮点数
tmp = dim//2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), "constant")
for i in range(dx):
    for j in range(dy):
        new_img[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
plt.figure(1)
plt.imshow(new_img.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

#3.运用Sobel算子求梯度进行边缘检测
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#便于展示各方向的梯度 所以设置x方向 y方向 总方向的梯度
slope_x, slope_y, slope = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)
#步骤与高斯滤波基本相同 接下来进行padding和遍历
img_pad = np.pad(img, ((1, 1), (1, 1)), "constant") #半径1.5 向下取1
for i in range(dx):
    for j in range(dy):
        slope_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
        slope_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
        slope[i, j] = np.sqrt(slope_x[i, j]**2 + slope_y[i, j]**2)
#为了防止分母为0 假设分母为0的值为0.00000001
slope_x[slope_x == 0] = 0.0000001
angle = slope_y/slope_x
plt.figure(2)
plt.imshow(slope.astype(np.uint8), cmap='gray')
plt.axis('off')

#3非极大值抑制
restrain = np.zeros([dx, dy])
#没有做padding 所以不检测边缘点 -1
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = slope[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1] #计算第一个点梯度
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1] #计算第二个点梯度
            if not (slope[i, j] > num_1 and slope[i, j] > num_2): #如果没有同时比两个点大 就抹去
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (slope[i, j] > num_1 and slope[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (slope[i, j] > num_1 and slope[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (slope[i, j] > num_1 and slope[i, j] > num_2):
                flag = False
        #如果flag为true则表示梯度比其它两个点大 所以保留
        if flag:
            restrain[i, j] = slope[i, j]
plt.figure(3)
plt.imshow(restrain.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4、双阈值检测
lower_boundary = slope.mean() * 0.5
high_boundary = lower_boundary * 3  # 高阈值永远是低阈值的三倍
#创建一个栈来存储
zhan = []
for i in range(1, restrain.shape[0] - 1):  # 同样不考虑边缘
    for j in range(1, restrain.shape[1] - 1):
        if restrain[i, j] >= high_boundary:  # 强边缘
            restrain[i, j] = 255
            zhan.append([i, j])
        elif restrain[i, j] <= lower_boundary:  # 非边缘
            restrain[i, j] = 0

#检验弱边缘点
while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    a = restrain[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    #判断每个临边是不是都是弱边缘 是的话就说明这个点不是噪声 连续 变为强边缘
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        restrain[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
        zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        restrain[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        restrain[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        restrain[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        restrain[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        restrain[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        restrain[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        restrain[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

#如果既不是0也不是255 就抑制
for i in range(restrain.shape[0]):
    for j in range(restrain.shape[1]):
        if restrain[i, j] != 0 and restrain[i, j] != 255:
            restrain[i, j] = 0
plt.figure(4)
plt.imshow(restrain.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值
plt.show()



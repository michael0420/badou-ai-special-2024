import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

#1. 读取图片+灰度化
img = cv2.imread('lenna.png',0)
sigma = 0.7
plt.figure(1)
plt.imshow(img.astype(np.uint8), cmap='gray')
plt.axis('off')
#2.进行高斯滤波
dim = 5 #高斯核大小
Guassian_fillter = np.zeros([dim,dim]) # 储存高斯核
tmp = [i-dim//2 for i in range(dim)] #[-2,-1,0,1,2]
#常数项
n2 = -1/2*sigma**2
n1 = 1/2*math.pi*sigma**2
#计算高斯核
for i in range(dim):
    for j in range(dim):
        Guassian_fillter[i,j] = n1 * math.exp(n2*tmp[i]**2+tmp[j]**2)
Guassian_fillter = Guassian_fillter / Guassian_fillter.sum() #高斯核归一化，保证输入输出图的亮度不变

#3.高斯平滑
h,w = img.shape
r = dim//2 #卷积核半径
img_new = np.zeros(img.shape)
img_pad = np.pad(img,((r,r),(r,r)),'constant') # 边缘填补,方便滤波器处理边缘像素点
#卷积
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_new[i,j] =np.sum(img_pad[i:i+dim,j:j+dim]*Guassian_fillter)

#4. 梯度计算(sobel算子)
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y =np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
tidu_x = np.zeros(img.shape)
tidu_y = np.zeros(img.shape)
img_tidu = np.zeros(img.shape)
img_pad = np.pad(img_new,((1,1),(1,1)),'constant')
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        tidu_x[i,j] =np.sum(img_pad[i:i+3,j:j+3] * sobel_x)
        tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3] * sobel_y)
        img_tidu[i,j] = np.sqrt(tidu_x[i,j]**2+tidu_y[i,j]**2)
#除数不能为0
tidu_x[tidu_x == 0] = 0.00000001
angle = tidu_y/tidu_x #tan = y/x
plt.figure(3)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

#5. 非极大值抑制
img_yizhi = np.zeros(img.shape)
for i in range(1,img_yizhi.shape[0]-1):
    for j in range(1,img_yizhi.shape[1]-1):
        flag = True # 标记是否消除
        temp = img_tidu[i-1:i+2,j-1:j+2]#领域点矩阵
        #x方向
        if angle[i,j] < -1 :
            num1 = (temp[0,1]-temp[0,0])/angle[i,j] + temp[0,1]
            num2 = (temp[2,1]-temp[2,2])/angle[i,j] + temp[2,1]
            if not img_tidu[i,j] > num1 and img_tidu[i,j] > num2:
                flag = False
        elif angle[i,j] > 1 :
            num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not img_tidu[i, j] > num1 and img_tidu[i, j] > num2:
                flag = False
        #y方向
        elif angle[i,j] > 0 :
            num1 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            num2 = (temp[0,2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            if not img_tidu[i, j] > num1 and img_tidu[i, j] > num2:
                flag = False
        elif angle[i,j] < 0:
            num1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1,2]
            if not img_tidu[i, j] > num1 and img_tidu[i, j] > num2:
                flag = False
        if flag:
            img_yizhi[i,j] = img_tidu[i,j]

 # 6 双阈值检测  连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
zhan = []
lower = img_tidu.mean()*0.5
higher = lower * 3
#保证zhan里面的每个点都有8领域点
for i in range(1,img_yizhi.shape[0]-1):
    for j in range(1,img_yizhi.shape[1]-1):
        if img_yizhi[i,j] > higher:
            img_yizhi[i,j] = 255
            zhan.append([i,j])
        elif img_yizhi[i,j] < lower:
            img_yizhi[i,j] = 0
print("shape:",img.shape)
while not len(zhan) == 0:
    temp_1,temp_2 = zhan.pop()
    a = img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
    ##看8邻域中是否有点是强边缘
    if (a[0, 0] < higher) and (a[0, 0] > lower):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 边缘
        zhan.append([temp_1 - 1, temp_2 - 1])  #进栈
    if (a[0, 1] < higher) and (a[0, 1] > lower):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < higher) and (a[0, 2] > lower):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < higher) and (a[1, 0] > lower):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] <higher) and (a[1, 2] > lower):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < higher) and (a[2, 0] > lower):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < higher) and (a[2, 1] > lower):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < higher) and (a[2, 2] > lower):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])
#去除孤点噪声
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i,j] != 0 and img_yizhi[i,j] != 255:
            img_yizhi[i,j] = 0
# 绘图
    plt.figure(2)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
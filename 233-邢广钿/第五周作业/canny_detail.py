import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


# 1、图片读取
img = plt.imread("../lenna.png")
# 2、灰度化
# img = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
img = img*255
img = img.mean(axis=-1)
print(img)


# 3、高斯平滑
sigma = 0.5
dim = 5 #高斯核尺寸
gs_filter = np.zeros([dim,dim])
#生成一个序列 [-2, -1, 0, 1, 2] 表示每个点的位置相对中心点的偏移
temp = [i-dim//2 for i in range(dim)]
# 计算高斯核
n1 = 1/(2*math.pi*sigma**2) #高斯核计算公式提取出来计算的一个公参 (为了方便下面计算)
n2 = -1/(2*sigma**2) #高斯核计算公式提取出来计算的一个公参 (为了方便下面计算)
for i in range(dim):
    for j in range(dim):
        gs_filter[i,j] = n1*math.exp(n2*(temp[i]**2+temp[j]**2))
gs_filter = gs_filter / gs_filter.sum() #将计算出来高斯核的值归一化
#对原图片进行边缘补齐
dx,dy= img.shape
img_new = np.zeros(img.shape)
pad_size = dim//2
img_pad = np.pad(img,((pad_size,pad_size),(pad_size,pad_size)),"constant")
#对原图进行高斯平滑计算
for x in range(dx):
    for y in range(dy):
        img_new[x,y] = np.sum(img_pad[x:x+dim,y:y+dim]*gs_filter)


# 4、通过sobel计算梯度值
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
pad_size = 1
img_tidu_x = np.zeros(img.shape)
img_tidu_y = np.zeros(img.shape)
img_tidu = np.zeros(img.shape)
img_pad = np.pad(img,((pad_size,pad_size),(pad_size,pad_size)),"constant")
for x in range(dx):
    for y in range(dy):
        img_tidu_x[x,y] = np.sum(img_pad[x:x+3,y:y+3]*sobel_x)
        img_tidu_y[x,y] = np.sum(img_pad[x:x+3,y:y+3]*sobel_y)
        # 计算梯度值
        img_tidu[x,y] = math.sqrt(img_tidu_x[x,y]**2+img_tidu_y[x,y]**2)
img_tidu_x[img_tidu_x == 0] =0.00000001
# 计算梯度角度
angle = img_tidu_y/img_tidu_x
img_tidu.astype(np.uint8)


# 5、对梯度值进行极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for x in range(1,dx-1):
    for y in range(1,dy-1):
        flag = True
        tidu_8 = img_tidu[x-1:x+2,y-1:y+2]
        agl = angle[x,y]
        if agl <= -1:
            num1 = (tidu_8[0,1] - tidu_8[0,0])/agl + tidu_8[0,1]
            num2 = (tidu_8[2,1] - tidu_8[2,2])/agl + tidu_8[2,1]
            if not (img_tidu[x, y] > num1 and img_tidu[x, y] > num2):
                flag = False
        elif agl >= 1:
            num1 = (tidu_8[0, 2] - tidu_8[0, 1]) / agl + tidu_8[0, 1]
            num2 = (tidu_8[2, 0] - tidu_8[2, 1]) / agl + tidu_8[2, 1]
            if not (img_tidu[x, y] > num1 and img_tidu[x, y] > num2):
                flag = False
        elif agl > 0:
            num1 = (tidu_8[0, 2] - tidu_8[1, 2]) * agl + tidu_8[1, 2]
            num2 = (tidu_8[2, 0] - tidu_8[1, 0]) * agl + tidu_8[1, 0]
            if not (img_tidu[x, y] > num1 and img_tidu[x, y] > num2):
                flag = False
        elif agl < 0:
            num1 = (tidu_8[1, 0] - tidu_8[0, 0]) * agl + tidu_8[1, 0]
            num2 = (tidu_8[1, 2] - tidu_8[2, 2]) * agl + tidu_8[2, 2]
            if not (img_tidu[x, y] > num1 and img_tidu[x, y] > num2):
                flag = False
        if flag:
            img_yizhi[x, y] = img_tidu[x, y]

# 6、双阈值检测
lower_boundary = img_tidu.mean() * 0.5 #梯度矩阵均值的0.5倍 可自定义非固定值
high_boundary = lower_boundary * 3
zhan = [] #记录强边缘点
for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  # 舍
            img_yizhi[i, j] = 0

while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop()  # 出栈
    # 获取强边缘八邻域的点
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    # 判断八邻域的点是否在阈值范围内 是则标记为强边缘 并且进栈会再次以相同逻辑判断新增的点
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255
        zhan.append([temp_1 - 1, temp_2 - 1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

# 绘图
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()
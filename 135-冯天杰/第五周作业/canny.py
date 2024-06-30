import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.color import rgb2gray

img = plt.imread("lenna.png")
img = rgb2gray(img)*255
sigma = 0.5  # 指定高斯参数

# 配置高斯核
dim = 5
gaussion_filter = np.zeros([dim, dim])

tmp = [i - dim//2 for i in range(dim)]
for i in range(dim):
    for j in range(dim):
        gaussion_filter[i, j] = (1/(2*math.pi*sigma**2))*math.exp(-(tmp[i]**2+tmp[j]**2)/(2*sigma**2))

gaussion_filter = gaussion_filter/gaussion_filter.sum()
# print(gaussion_filter)

# 对图像进行高斯平滑
x,y = img.shape
tmp = dim//2
img_pad =np.pad (img,((tmp,tmp),(tmp,tmp)),"constant")
img_gaussion = np.zeros(img.shape)
for i in range(x):
    for j in range(y):
        img_gaussion[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*gaussion_filter)

plt.imshow(img_gaussion.astype(np.uint8),cmap="gray")

# 利用sobel提取边缘
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

img_sobel_x = np.zeros(img_gaussion.shape)
img_sobel_y = np.zeros(img_gaussion.shape)
img_sobel_xy =np.zeros(img_gaussion.shape)
img_pad = np.pad(img_gaussion,((1,1),(1,1)),"constant")
for i in range(x):
    for j in range(y):
        img_sobel_x[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sobel_x)
        img_sobel_y[i,j]=np.sum(img_pad[i:i+3,j:j+3]*sobel_y)
        img_sobel_xy[i,j]=np.sqrt(img_sobel_x[i,j]**2+img_sobel_y[i,j]**2)

plt.imshow(img_sobel_xy.astype(np.uint8),cmap="gray")

# 非极大值抑制
img_sobel_x[img_sobel_x == 0] == 0.000000001
tan = img_sobel_y/img_sobel_x

img_yizhi = np.zeros(img_sobel_xy.shape)
for i in range(1,x-1):
    for j in range(1,y-1):
        flag = True  #先将所有的点变为ture，后面判断如果被抑制，就变为false
        tamp = img_sobel_xy[i-1:i+2,j-1:j+2]
        # 利用差值法求梯度方向上邻近点的差值，并比较其与之的大小
        if tan[i,j] >= 1:
            tamp1 = (1-1/tan[i,j])*tamp[0,1]+1/tan[i,j]*tamp[0,2]
            tamp2 = (1-1/tan[i,j])*tamp[2,1]+1/tan[i,j]*tamp[2,0]
            if not (img_sobel_xy[i,j] > tamp1 and img_sobel_xy[i,j] > tamp2):
                flag = False
        elif tan[i,j] <= -1:
            tamp1 = (1-(-1/tan[i,j]))*tamp[0,1]+(-1/tan[i,j]*tamp[0,0])
            tamp2 = (1-(-1/tan[i,j]))*tamp[2,1]+(-1/tan[i,j]*tamp[2,2])
            if not (img_sobel_xy[i,j] > tamp1 and img_sobel_xy[i,j] > tamp2):
                flag = False
        elif 0 <tan[i,j] < 1:
            tamp1 = (1-tan[i,j])*tamp[1,2]+tan[i,j]*tamp[0,2]
            tamp2 = (1-tan[i,j])*tamp[1,0]+tan[i,j]*tamp[2,0]
            if not (img_sobel_xy[i,j] > tamp1 and img_sobel_xy[i,j] > tamp2):
                flag = False
        elif -1 <tan[i,j] < 0:
            tamp1 = (1-(-tan[i,j]))*tamp[1,0]+(-tan[i,j]*tamp[0,0])
            tamp2 = (1-(-tan[i,j]))*tamp[1,2]+(-tan[i,j]*tamp[2,2])
            if not (img_sobel_xy[i,j] > tamp1 and img_sobel_xy[i,j] > tamp2):
                flag = False
        if flag:
            img_yizhi[i,j] = img_sobel_xy[i,j]

# 双值域（设置上下值，高于最大的的变为255，即为边缘，低于最小的变为0。再调整边缘点的领域点为255，其他的为0）

low = img_sobel_xy.mean()*0.5
high = low *3
zhan = []
for i in range(x):
    for j in range(y):
        if img_yizhi[i,j] >= high:
            img_yizhi[i,j] = 255
            zhan.append([i,j])
        elif img_yizhi[i,j]<=low:
            img_yizhi[i,j]= 0

while not len(zhan) ==0:
    tamp1,tamp2 = zhan.pop()
    a = img_yizhi[tamp1-1:tamp1+2,tamp2-1:tamp2+2]
    if low <a[0,0] <high:
        img_yizhi[tamp1-1, tamp2-1] =255
        zhan.append([tamp1-1,tamp2-1])
    if low <a[0,1] <high:
        img_yizhi[tamp1-1,tamp2] =255
        zhan.append([tamp1-1,tamp2])
    if low <a[0,2] <high:
        img_yizhi[tamp1-1,tamp2+1] =255
        zhan.append([tamp1-1,tamp2+1])
    if low <a[1,0] <high:
        img_yizhi[tamp1,tamp2-1] =255
        zhan.append([tamp1,tamp2-1])
    if low <a[1,2] <high:
        img_yizhi[tamp1,tamp2+1] =255
        zhan.append([tamp1,tamp2+1])
    if low <a[2,0] <high:
        img_yizhi[tamp1+1,tamp2-1] =255
        zhan.append([tamp1+1,tamp2-1])
    if low <a[2,1] <high:
        img_yizhi[tamp1+1,tamp2] =255
        zhan.append([tamp1+1,tamp2])
    if low <a[2,2] <high:
        img_yizhi[tamp1+1,tamp2+1] =255
        zhan.append([tamp1+1,tamp2+1])

# 去噪声
for i in range(x):
    for j in range(y):
        if  img_yizhi[i,j] !=255 and img_yizhi[i,j] != 0:
            img_yizhi[i,j]=0


plt.imshow(img_yizhi.astype(np.uint8),cmap="gray")
plt.show()

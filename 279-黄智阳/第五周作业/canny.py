import matplotlib.pyplot as plt
import numpy as np
import math

pic_path='lenna.png'
img=plt.imread(pic_path)
print(img)
if pic_path[-4:]=='.png':
    img = img*255
img = img.mean(axis=-1)

# 1,高斯平滑

sigma=0.5
dim = 5
Gaussian_filter = np.zeros([dim,dim])
tmp = [i-dim//2 for i in range(dim)]
n1 = 1/(2*math.pi*sigma**2)
n2 = 1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
dx, dy = img.shape
img_new = np.zeros(img.shape)
tmp = dim//2
img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)
plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

# 2,求梯度，使用sobel
sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
img_tidu_x = np.zeros(img_new.shape)
img_tidu_y = np.zeros(img_new.shape)
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new,((1,1),(1,1)),'constant')
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
        img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)
        img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2+img_tidu_y[i,j]**2)
img_tidu_x[img_tidu_x==0] = 0.00000001
angle = img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

# 3,非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1,dx-1):
    for j in range(1,dy-1):
        flag=True
        temp = img_tidu[i-1:i+2,j-1:j+2]
        if angle[i,j]<=-1:
            num_1 = (temp[0,1]-temp[0,0])/angle[i,j]+temp[0,1]
            num_2 = (temp[2,1]-temp[2,2])/angle[i,j]+temp[2,1]
            if not (img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2):
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
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 4双阀值检测
lower_boundary = img_tidu.mean()*0.5
high_boundary = lower_boundary*3
zhan = []
for i in range(1,img_yizhi.shape[0]-1):
    for j in range(1,img_yizhi.shape[1]-1):
        if img_yizhi[i,j]>=high_boundary:
            img_yizhi[i,j]=255
            zhan.append([i,j])
        elif img_yizhi[i,j]<=lower_boundary:
            img_yizhi[i,j]=0

while not len(zhan)==0:
    temp_1,temp_2 = zhan.pop()
    a = img_yizhi[temp_1-1:temp_1+2,temp_2-1:temp_2+2]
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
        zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
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

plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.show()
import numpy
import matplotlib.pyplot as plt
import math
# 梯度线
#
import numpy as np

img_path = "../../lenna.png"
img = plt.imread(img_path)
if img_path[-4:] == '.png':
    img = img * 255
gray_img = img.mean(axis=-1)
plt.figure(1)
plt.imshow(gray_img.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

# 高斯滤波
sigma = 0.5
dim = 5
# 创建高斯过滤器
Gaussian_filter = np.zeros((dim, dim))
# //符号为整除符号
tmp = [i - dim // 2 for i in range(dim)]
print("tmp: ", tmp)

# 计算高斯函数的公式中，提取常数部分
n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
n2 = -1 / (2 * sigma ** 2)
# 计算5*5每个位置的高斯值
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
# 高斯滤波器的归一化
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()

# 开始高斯滤波
# 存储滤波后的新图片
new_img = np.zeros((gray_img.shape))
tmp = dim // 2

# 在灰度图周围添加一圈像素
img_pad = np.pad(gray_img, ((tmp, tmp), (tmp, tmp)), 'constant')
# print(gray_img)
print("img_pad: ",img_pad.shape)
# 获取原图片x和y
dx, dy = gray_img.shape
for i in range(dx):
    for j in range(dy):
        new_img[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)

plt.figure(2)
plt.imshow(new_img.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

# 计算梯度,检测图像中的水平、垂直和对角边缘
#梯度代表了图像像素的变化率，梯度越大表示变化越激烈，
#所以在边缘位置梯度值较高
#sobel算子的结果，就是对应像素的梯度值
soble_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
soble_kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

img_tidu_x = np.zeros(new_img.shape)
img_tidu_y = np.zeros(new_img.shape)
img_tidu = np.zeros(new_img.shape)

#边缘填充
img_pad = np.pad(new_img,(1,1),'constant')
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3] * soble_kernel_x)
        img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3] * soble_kernel_y)
        #sobel边缘检测使用根号下x方+y方（也就是计算梯度大小）
        img_tidu[i,j] = np.sqrt(img_tidu_x[i,j] ** 2 + img_tidu_y[i,j] ** 2)

#img_tidu_x == 0 是一种广播操作，它会对矩阵 img_tidu_x 的每个元素进行逐个比较，
# 判断元素是否等于0。如果元素等于0，则对应位置的元素值为 True，否则为 False。
#在这种情况下，我们可以将 img_tidu_x 视为一个整体，
# 并使用 img_tidu_x == 0 来表示对整个矩阵进行元素级别的比较操作。
# 因此，img_tidu_x == 0 可以得到一个与 img_tidu_x 具有相同形状的布尔矩阵，指示了哪些位置的元素等于0。
img_tidu_x[img_tidu_x == 0] = 0.00000001

angle = img_tidu_y / img_tidu_x
img_tidu_uint8 = img_tidu.astype(np.uint8)

plt.figure(3)
plt.imshow(img_tidu_uint8, cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
plt.axis('off')

#非极大值抑制
print(img_tidu)
print("-------------------------")
img_yizhi = np.zeros(img_tidu.shape)

for i in range(1,dx - 1):
    for j in range(1,dy -1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
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
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
#
# #
#双阈值检测
low_b = img_tidu.mean()*0.5
high_b = low_b * 3
zhan= []
for i in range(1,img_yizhi.shape[0]-1):
    for j in range(1,img_yizhi.shape[1]-1):
        if img_yizhi[i,j]>=high_b:
            img_yizhi[i,j] = 255
            zhan.append([i,j])
        elif img_yizhi[i,j]<=low_b:
            img_yizhi[i, j] = 0
plt.figure(5)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')
# #进入栈的都是大于等于高阈值，接下来需要检测这个点周围八像素是否在高领域和低领域之间
while len(zhan) != 0:
    temp1, temp2 = zhan.pop()
    a = img_yizhi[temp1-1:temp1+2 , temp2-1:temp2+2]
    if(a[0,0] > low_b and a[0,0] < high_b):
        img_yizhi[temp1-1,temp2-1] = 255
        zhan.append([temp1-1,temp2-1])
    if(a[0,1] > low_b and a[0,1]< high_b):
        img_yizhi[temp1 - 1, temp2] = 255
        zhan.append([temp1 - 1, temp2])
    if (a[0, 2] > low_b and a[0, 2] < high_b):
        img_yizhi[temp1-1, temp2+1] = 255
        zhan.append([temp1 - 1, temp2 +1])
    if (a[1, 0] > low_b and a[1, 0] < high_b):
        img_yizhi[temp1, temp2-1] = 255
        zhan.append([temp1, temp2 -1])
    if (a[1, 2] > low_b and a[1, 2] < high_b):
        img_yizhi[temp1, temp2+1] = 255
        zhan.append([temp1, temp2 +1])
    if (a[2, 0] > low_b and a[2, 0] < high_b):
        img_yizhi[temp1+1, temp2-1] = 255
        zhan.append([temp1+1, temp2 -1])
    if (a[2, 1] > low_b and a[2, 1] < high_b):
        img_yizhi[temp1+1, temp2] = 255
        zhan.append([temp1+1, temp2])
    if (a[2, 2] > low_b and a[2, 2] < high_b):
        img_yizhi[temp1+1, temp2+1] = 255
        zhan.append([temp1+1, temp2+1])

#通过遍历所有高阈值点的周围像素，剩余的就是没有连接的孤立点，需要抑制
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i,j] !=0 and img_yizhi[i,j] !=255:
            img_yizhi[i,j] = 0


plt.figure(6)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值
plt.show()


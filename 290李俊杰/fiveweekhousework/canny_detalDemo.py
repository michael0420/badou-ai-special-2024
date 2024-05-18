'''
【第五周作业】
1.实现canny detail  
Canny边缘检测算法
1. 对图像进行灰度化
2. 对图像进行高斯滤波：
根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
可以有效滤去理想图像中叠加的高频噪声。
3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
4 对梯度幅值进行非极大值抑制
5 用双阈值算法检测和连接边缘
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

# 1. 对图像进行灰度化
img = cv2.imread("lenna.png", 0)
# print(img) #img的矩阵里面存储的是整数类型0-255

# 2. 对图像进行高斯滤波：
# 根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
# 可以有效滤去理想图像中叠加的高频噪声。
sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
# 高斯核尺寸(定义卷积核矩阵）5*5
dim = 5
Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
'''
dim=5时，生成tmp的结果[-2,-1,0,1,2]，
代表5*5邻域内各点与中心点的坐标差值，
这就是代入高斯方程中的x、y值
'''
tmp = []
for i in range(dim):
    tmp.append(i - dim // 2)
# print(tmp)
# 求高斯卷积盒矩阵内的每一个值，
# 套入公式G(x,y)=1/(2*π*sigma**2)*e**(-1*((x**2+y**2)/(2*sigma**2)))
# 套入公式计算出高斯核里面的每一个值，math.pi圆周率函数
# math.exp方法返回e的x次幂次方Ex其中e=2.718281... 是自然对数的基数
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i, j] = 1 / (2 * math.pi * sigma ** 2) * math.exp(
            -1 * ((tmp[i] ** 2 + tmp[j] ** 2) / (2 * sigma ** 2)))
# 对高斯核进行归一化处理
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
# print(Gaussian_filter)


# 建立一个空矩阵存储平滑之后的图像，zeros函数得到的是浮点型数据
img_new = np.zeros(img.shape)
"""
np.pad()用来在numpy数组的边缘进行数值填充，例如CNN网络常用的padding操作
np.pad(array，pad_width，mode， ** kwargs)　　  # 返回填充后的numpy数组
参数：
array：要填充的numpy数组【要对谁进行填充】
pad_width：每个轴要填充的数据的数目【每个维度前、后各要填充多少个数据】
第一个括号内代表（上，下），第二个代表(左，右)，如果只有一个括号就是（上下，左右）
mode：填充的方式【采用哪种方式填充】
"""
# 要对原图每一个像素使用5*5的卷积核必须对图片边缘进行填充，每个边缘向外扩大两行
# 直接通过计算卷积核的中心离边缘的距离算出原图边缘要填充的数值
# print(tmp)
img_pad = np.pad(img, (dim // 2, dim // 2), 'constant')
# print(img_pad)
img_h, img_w = img.shape
# 使用高斯核对原图每一个像素进行卷积
# 注意，要用填充后的原图img_pad才能计算出原图img_new上每一个像素卷积后的值

for i in range(img_h):
    for j in range(img_w):
        img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
# print(img_new)
# 此时的img_new是255的浮点型数据，强制类型转换才可以
# cv2.imshow("img_new",img_new.astype(np.uint8))
# cv2.waitKey(0)

# 3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
# 使用sobel检测出图片横向和纵向的边缘
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 检测横向边缘的sobel
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # 检测纵向边缘的sobel
img_sobx = np.zeros(img_new.shape)  # 存储横向边缘的空矩阵
img_soby = np.zeros(img_new.shape)  # 存储纵向边缘的sobel
img_sobxy = np.zeros(img_new.shape)  # 存储横纵一起边缘的sobel
img_new_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
for i in range(img_h):
    for j in range(img_w):
        img_sobx[i, j] = np.sum(img_new_pad[i:i + 3, j:j + 3] * sobel_x)
        img_soby[i, j] = np.sum(img_new_pad[i:i + 3, j:j + 3] * sobel_y)
        img_sobxy[i, j] = np.sqrt(img_sobx[i, j] ** 2 + img_soby[i, j] ** 2)
if img_sobx[i, j] == 0:
    img_sobx[i, j] = 0.00000001  # 防止分母为0计算不了
# 求出tan值得到梯度所在象限
tanyx = img_soby / img_sobx
# cv2.imshow("img_sobx",img_sobx.astype(np.uint8))
# cv2.imshow("img_soby",img_soby.astype(np.uint8))
# cv2.imshow("img_sobxy",img_sobxy.astype(np.uint8))
# cv2.waitKey(0)

# 4 对梯度幅值进行非极大值抑制
#
img_yz = np.zeros(img_sobxy.shape)
for i in range(1, img_h - 1):
    for j in range(1, img_w - 1):
        flag = True  # 在8邻域内是否要抹去做个标记
        temp = img_sobxy[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
        if tanyx[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (temp[0, 1] - temp[0, 0]) / tanyx[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / tanyx[i, j] + temp[2, 1]
            if not (img_sobxy[i, j] > num_1 and img_sobxy[i, j] > num_2):
                flag = False
        elif tanyx[i, j] >= 1:
            num_1 = (temp[0, 2] - temp[0, 1]) / tanyx[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / tanyx[i, j] + temp[2, 1]
            if not (img_sobxy[i, j] > num_1 and img_sobxy[i, j] > num_2):
                flag = False
        elif tanyx[i, j] > 0:
            num_1 = (temp[0, 2] - temp[1, 2]) * tanyx[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * tanyx[i, j] + temp[1, 0]
            if not (img_sobxy[i, j] > num_1 and img_sobxy[i, j] > num_2):
                flag = False
        elif tanyx[i, j] < 0:
            num_1 = (temp[1, 0] - temp[0, 0]) * tanyx[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * tanyx[i, j] + temp[1, 2]
            if not (img_sobxy[i, j] > num_1 and img_sobxy[i, j] > num_2):
                flag = False
        if flag:
            img_yz[i, j] = img_sobxy[i, j]
# cv2.imshow("img_yz",img_yz.astype(np.uint8))
# cv2.waitKey(0)

# 5 用双阈值算法检测和连接边缘
# 设置高低阈值 （根据真实场景人为设置没有固定值）
lower_boundary = img_sobxy.mean() * 0.5
high_boundary = lower_boundary * 3

zhan = []
# 遍历非极大值抑制后的图中的所有像素，对高阈值以上的变成255(强边缘)
# ，低阈值以下的变成0(弱边缘—)
for i in range(1, img_yz.shape[0] - 1):  # 外圈不考虑了
    for j in range(1, img_yz.shape[1] - 1):
        if img_yz[i, j] >= high_boundary:  # 取，一定是边的点
            img_yz[i, j] = 255
            zhan.append([i, j])  # 将强边缘点的坐标存储起来
        elif img_yz[i, j] <= lower_boundary:  # 舍
            img_yz[i, j] = 0
    # 对于处于高低阈值之间的像素，再次进行判断，若这些像素的八邻域中有像素值
    # 为255的，则将此像素与强边缘连接也就是设置为强边缘(255)
    # 根据上诉情况可以反向推导出，只要强边缘像素周围8邻域有
    # 这种像素值存在就可以将其设置为255
while not len(zhan) == 0:
    # 出栈 把每个强边缘点的坐标取出来将其x，y坐标分别放入两个变量中
    temp_1, temp_2 = zhan.pop()
    # 根据这个强边缘点得到周围8邻域矩阵（也包括自己）
    a = img_yz[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
    # 对矩阵中的每个像素值判断，如果有处于高低阈值之间的像素值
    # 将其像素值变为255
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yz[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
        zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yz[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yz[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yz[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yz[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yz[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yz[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yz[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

# 此时图中的强边缘点已经找完，剩下的像素点还有数值的就可以变成0抑制掉
for i in range(img_yz.shape[0]):
    for j in range(img_yz.shape[1]):
        if img_yz[i, j] != 0 and img_yz[i, j] != 255:
            img_yz[i, j] = 0
cv2.imshow("img_yz", img_yz.astype(np.uint8))
cv2.waitKey(0)

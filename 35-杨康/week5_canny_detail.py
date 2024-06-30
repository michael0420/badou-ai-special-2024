import numpy as np
import matplotlib.pyplot as plt
import math
"""
1. 对图像进行灰度化
2. 对图像进行高斯滤波：
根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样
可以有效滤去理想图像中叠加的高频噪声。
3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
4 对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点
所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
5 用双阈值算法检测和连接边缘
"""
if __name__ == '__main__':
    # 1.对图像进行灰度化
    img = plt.imread('lenna.png')*255    #.png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    img_gray = img.mean(axis=-1)         #三通道均值法进行灰度化
    #2. 对图像进行高斯滤波
    sigma = 0.5  #设置高斯核sigma值，可变
    dim = 5      #设置高斯核大小
    temp = [i-dim//2 for i in range(dim)]
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    gaussian_filter = np.zeros([dim,dim])  #用来存储高斯核数据
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i,j] = n1*math.exp(n2*(temp[i]**2+temp[j]**2))
    print(gaussian_filter)
    gaussian_filter = gaussian_filter/gaussian_filter.sum()
    pad = dim//2
    h, w = img_gray.shape
    img_pad = np.pad(img_gray,((pad,pad),(pad,pad)),'constant')  #对原灰度图进行边缘填充
    img_gaussian = np.zeros(img_gray.shape)
    for i in range(h):
        for j in range(w):
            img_gaussian[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*gaussian_filter)
    print(img_gaussian)
    plt.figure(1)
    plt.imshow(img_gaussian.astype(np.uint8),cmap='gray')

    #3.检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）
    # 用Sobel算子检测高斯滤波后的图像边缘，获得x,y方向梯度图
    Sobel_kernel_x = np.array([[-1,0,1],
                               [-2,0,2],
                               [-1,0,1]])
    Sobel_kernel_y = np.array([[1,2,1],
                               [0,0,0],
                               [-1,-2,-1]])
    dx,dy = img_gaussian.shape
    img_sobel_x = np.zeros(img_gaussian.shape)
    img_sobel_y = np.zeros(img_gaussian.shape)
    img_sobel = np.zeros(img_gaussian.shape)
    img_pad = np.pad(img_gaussian,((1,1),(1,1)),mode='constant')
    for i in range(dx):
        for j in range(dy):
            img_sobel_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*Sobel_kernel_x)
            img_sobel_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * Sobel_kernel_y)
            img_sobel[i,j] = np.sqrt(img_sobel_x[i,j]**2+img_sobel_y[i, j]**2)
    print(img_sobel)
    plt.figure(2)
    plt.imshow(img_sobel.astype(np.uint8), cmap='gray')

    # 4.对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
    img_sobel_x[img_sobel_x == 0] = 0.00000001
    angle = img_sobel_y/img_sobel_x
    img_yizhi = np.zeros(img_sobel.shape)
    for i in range(1,img_sobel.shape[0]-1):
        for j in range(1,img_sobel.shape[1]-1):
            flag = True
            grad = img_sobel[i - 1:i + 2, j - 1:j + 2]  #8邻域像素梯度矩阵
            #用线性插值法计算虚拟点像素梯度，抑制非极大值
            if angle[i,j] <= -1:
                temp1 = grad[0,0]/(-1*angle[i,j]) + grad[0,1]*(1-1/(-1*angle[i,j]))   #angle为负数时要取反运算
                temp2 = grad[2,1]*(1-1/(-1*angle[i,j])) + grad[2,2]/(-1*angle[i,j])
                if not (img_sobel[i, j] > temp1 and img_sobel[i, j] > temp2):
                    flag = False
            elif angle[i,j] >= 1:
                temp1 = grad[0,1]*(1-1/angle[i,j]) + grad[0,2]/angle[i,j]
                temp2 = grad[2,0]/angle[i,j] + grad[2,1]*(1-1/angle[i,j])
                if not (img_sobel[i, j] > temp1 and img_sobel[i, j] > temp2):
                    flag = False
            elif angle[i,j] < 0:
                temp1 = grad[0,0]*(-1*angle[i,j]) + grad[1,0]*(1-(-1*angle[i,j]))    #angle为负数时要取反运算
                temp2 = grad[1,2]*(1-(-1*angle[i,j])) + grad[2,2]*(-1*angle[i,j])
                if not (img_sobel[i, j] > temp1 and img_sobel[i, j] > temp2):
                    flag = False
            elif angle[i,j] >0:
                temp1 = grad[1,0]*(1-angle[i,j]) + grad[2,0]*angle[i,j]
                temp2 = grad[0,2]*angle[i,j] + grad[1,2]*(1-angle[i,j])
                if not (img_sobel[i, j] > temp1 and img_sobel[i, j] > temp2):
                    flag = False
            if flag:
                img_yizhi[i,j] = img_sobel[i,j]
    print(img_yizhi)
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')

    #5.用双阈值算法检测和连接边缘
    lowThreshold = img_yizhi.mean()  #低阈值
    highThreshold = lowThreshold*3   #高阈值
    zhan = []  #存储弱边缘坐标
    for i in range(1,img_yizhi.shape[0]-1):
        for j in range(1,img_yizhi.shape[1]-1):
            if img_yizhi[i,j] >= highThreshold:
                img_yizhi[i, j] = 255
            elif img_yizhi[i,j] <= lowThreshold:
                img_yizhi[i, j] = 0
            else:
                zhan.append([i,j])  #把弱边缘坐标进栈
    for i in range(len(zhan)):
        temp1,temp2 = zhan[i]
        if 255 in img_yizhi[temp1-1:temp1+2,temp2-1:temp2+2]:  #若弱边缘8邻域内有强边缘，保留该边缘
            img_yizhi[temp1,temp2] = 255
        else:
            img_yizhi[temp1, temp2] = 0
    print(img_yizhi)
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()




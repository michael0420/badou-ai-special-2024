import numpy as np
import matplotlib.pyplot as plt
import math
'''
 1 灰度图片
 2 噪声降低 - 高斯滤波
 3 计算梯度强度和方向 -边缘检测 sobel滤波
 4 非极大值抑制
 5 双阈值检测
 6 边缘跟踪
'''

def iterate_neighbors(matrix, row, col):
    # 定义8个可能的方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    # 获取矩阵的行数和列数
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0

    for dx,dy in directions:
        new_row, new_col = row + dx, col + dy
        # 检查新位置是否在矩阵范围内
        if 0 <= new_row < rows and 0 <= new_col < cols:
            yield [new_row, new_col]

if __name__ == "__main__" :
    img_name = "lenna.png"
    img = plt.imread(img_name)

    if img_name[-4:] == ".png":  #png 文件时，结果是一个元素类型为浮点数且范围在0到1之间的数组
        img = img * 255

    #均值灰度化
    img = img.mean(axis=-1)
    print("img=\n", img)
    # plt.subplot(2, 2, 1)
    # plt.imshow(img.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶

    #高斯滤波
    segma = 0.5  # 标准差
    dim = 5     # 高斯核尺寸
    Gaussian_filter = np.zeros([dim, dim])
    tmp = [i - dim //2 for i in range(dim)]
    print("tmp=", tmp)
    n1 = 1/(2 * math.pi * segma ** 2)
    n2 = -1/( 2* segma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp( n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
    print("高斯核=\n", Gaussian_filter)

    #平滑
    dx, dy = img.shape
    #补充图片边缘
    pad_size = dim // 2
    img_pad = np.pad(img,((pad_size, pad_size),(pad_size, pad_size)),'constant')
    img_new = np.zeros([dx, dy])  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim] * Gaussian_filter)

    plt.subplot(2, 2, 1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    print("img_new=", img_new)

    #sobel 梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i,j] = np.sqrt(img_tidu_x[i,j] ** 2 + img_tidu_y[i, j] ** 2)
    #被除数不为0
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x

    plt.subplot(2, 2, 2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')  # 此时的img_tidu是255的浮点型数据，强制类型转换才可以，gray灰阶

    #非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
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

    plt.subplot(2, 2, 3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 此时的img_yizhi是255的浮点型数据，强制类型转换才可以，gray灰阶

    #双阈值检测
    lower_boundary = img_tidu.mean() * 0.5  #
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    print("lower_boundary=%s,high_boundary=%s"%(lower_boundary, high_boundary))
    tx,ty = img_yizhi.shape
    zhan = []
    for i in range(tx - 1):
        for j in range(ty - 1):
            if img_yizhi[i,j] >= high_boundary:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i,j] <= high_boundary:
                img_yizhi[i, j] = 0

    # 边缘跟踪
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        for x, y in iterate_neighbors(img_yizhi, temp_1, temp_2):
            if (img_yizhi[x, y] < high_boundary) and (img_yizhi[x, y] > lower_boundary):
                img_yizhi[x, y] = 255  # 这个像素点标记为边缘
                zhan.append([x, y])  # 进栈

    #去掉其他点
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i,j] != 0 or img_yizhi[i,j] != 255:
                img_yizhi[i, j] == 0


    plt.subplot(2, 2, 4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 此时的img_yizhi是255的浮点型数据，强制类型转换才可以，gray灰阶

    plt.show()


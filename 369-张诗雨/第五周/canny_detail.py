import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)  # 读入值为0-1之间的浮点数
    if pic_path[-4:] == '.png':
        img = img * 255
    img = img.mean(axis=-1)  # 沿最后一维取平均值，灰度化

    # 1. 高斯平滑
    sigma = 0.5
    dim = 5     # 高斯核大小
    # （1）计算高斯核
    gaussian_filter = np.zeros([dim, dim])  # 储存高斯核
    n1 = 1/(2*math.pi*(sigma**2))   # 计算高斯参数
    n2 = -1/(2*(sigma**2))
    arr = [i-dim//2 for i in range(dim)]    # 生成序列-2,-1,0,1,2
    for i in range(dim):
        for j in range(dim):
            u = arr[i]
            v = arr[j]
            gaussian_filter[i, j] = n1*math.exp(n2*(u**2+v**2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    # （2）高斯平滑
    dx, dy = img.shape
    img_gaussian = np.zeros(img.shape)
    pad = dim // 2
    img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_gaussian[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*gaussian_filter)
    plt.figure(1)
    plt.imshow(img_gaussian.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off') # 关闭坐标刻度值

    # 2. 求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros([dx, dy])
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros([dx, dy])
    img_pad = np.pad(img_gaussian, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(sobel_kernel_x*img_pad[i:i+3, j:j+3])
            img_tidu_y[i, j] = np.sum(sobel_kernel_y*img_pad[i:i+3, j:j+3])
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.0000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3. 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True     # flag：是否为极大值
            temp = img_tidu[i-1:i+2, j-1:j+2]   # 8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法
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
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4. 双阈值检测
    low_boundary = img_tidu.mean() * 0.5
    high_boundary = low_boundary * 3
    zhan = []   # 存所有是边缘的点
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            if img_yizhi[i, j] >= high_boundary:
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= low_boundary:
                img_yizhi[i, j] = 0

    while len(zhan) > 0:
        x, y = zhan.pop()
        # 访问强边缘的8邻域，如果有弱边缘，将其标记为强边缘，进栈
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                if low_boundary < img_yizhi[x+i-1, y+j-1] < high_boundary:
                    img_yizhi[x+i-1, y+j-1] = 255
                    zhan.append([x+i-1, y+j-1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

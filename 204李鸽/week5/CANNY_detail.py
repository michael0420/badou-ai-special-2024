import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == '__main__':
    name = 'lenna.png'
    img = plt.imread(name)
    print('img', img)
    if name[-4:] == '.png':    # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算（优化）
        img = img * 255        # 还是浮点数类型
    img_grey = img.mean(axis=-1)     # 取均值的方法进行灰度化（非color_based,也可用其他灰度化方法）

    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5     # 高斯核函数的，标准差，可调
    dim = 5         # 高斯卷积核的尺寸
    gaussian_filter = np.zeros([dim, dim])     # 存储高斯核，这是数组不是列表了
    tmp = [i - dim//2 for i in range(dim)]     # 生成一个中心对称的5*5序列，得到[-2,-1,0,1,2]
    n1 = 1 / (2 * math.pi * sigma ** 2)        # math.pi即Π，3.1415926
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):                   # e是常数,math.e  math.exp(x)表示e的x次方
            gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()     # 归一化，可有可无
    h, w = img_grey.shape
    img_new = np.zeros(img_grey.shape)     # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    r = dim // 2     #半径为2
    img_pad = np.pad(img_grey, ((r, r), (r, r)), 'constant')   # 边缘填补
    for i in range(h):
        for j in range(w):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * gaussian_filter)
    plt.figure(1)
    # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')      # 关闭坐标轴的显示

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_sobel_x = np.zeros(img_new.shape)
    img_sobel_y = np.zeros([h, w])
    img_sobel = np.zeros(img_new.shape)
    img_pad2 = np.pad(img_new, ((1, 1),(1, 1)), 'constant')
    for i in range(h):
        for j in range(w):
            img_sobel_x[i, j] = np.sum(img_pad2[i:i + 3, j:j + 3] * sobel_x)
            img_sobel_y[i, j] = np.sum(img_pad2[i:i + 3, j:j + 3] * sobel_y)
            img_sobel[i, j] = np.sqrt(img_sobel_x[i, j] ** 2 + img_sobel_y[i, j] ** 2)
    img_sobel_x[img_sobel_x == 0] = 0.00000001
    tan_angle = img_sobel_y / img_sobel_x
    plt.figure(2)
    plt.imshow(img_sobel.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_sobel.shape)
    for i in range(1, h-1):            # 不是range(h),即取0-（h-1),因为一会要以（i，j）为中心算八邻域
        for j in range(1, w-1):
            flag = True      # true就不抑制了
            tmp8 = img_sobel[i-1:i+2, j-1:j+2]
            if tan_angle[i, j] <= -1:          # 根据tan值的取值范围
                num_1 = (tmp8[0, 1] - tmp8[0, 0]) / tan_angle[i, j] + tmp8[0, 1]  # 求交点1的像素值
                num_2 = (tmp8[2, 1] - tmp8[2, 2]) / tan_angle[i, j] + tmp8[2, 1]  # 求交点2的像素值
                if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                    # 如果当前像素的梯度强度与另外两个像素相比最大，则该像素点保留为边缘点
                    # 如果不同时大于他俩，就要被抑制了
                    flag = False    # 要被抑制了
            elif tan_angle[i, j] >= 1:
                num_1 = (tmp8[0, 2] - tmp8[0, 1]) / tan_angle[i, j] + tmp8[0, 1]
                num_2 = (tmp8[2, 0] - tmp8[2, 1]) / tan_angle[i, j] + tmp8[2, 1]
                if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                    flag = False
            elif tan_angle[i, j] > 0:
                num_1 = (tmp8[0, 2] - tmp8[1, 2]) * tan_angle[i, j] + tmp8[1, 2]
                num_2 = (tmp8[2, 0] - tmp8[1, 0]) * tan_angle[i, j] + tmp8[1, 0]
                if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                    flag = False
            elif tan_angle[i, j] < 0:
                num_1 = (tmp8[1, 0] - tmp8[0, 0]) * tan_angle[i, j] + tmp8[1, 0]
                num_2 = (tmp8[1, 2] - tmp8[2, 2]) * tan_angle[i, j] + tmp8[1, 2]
                if not (img_sobel[i, j] > num_1 and img_sobel[i, j] > num_2):
                    flag = False
            if flag:     # 如果flag成立，
                img_yizhi[i, j] = img_sobel[i, j]      # 又过滤掉一部分点
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4.双阈值检测，看看这些目前是边缘的点，周围8邻域有没有可能也是边缘的点与他相连，进栈
    low_boundary = img_sobel.mean() * 0.5     # 随便定的
    high_boundary = low_boundary * 3    # 随便定的
    on_the_line = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                on_the_line.append([i, j])
            elif img_yizhi[i, j] <= low_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(on_the_line) == 0:
        m, n = on_the_line.pop()  # 最后一个元素出栈，循环，直到没有元素
        a = img_yizhi[m-1:m+2, n-1:n+2]        # 这些强边缘点的八邻域
        if (a[0, 0] < high_boundary) and (a[0, 0] > low_boundary):
            img_yizhi[m-1, n-1] = 255  # 这个像素点标记为模棱两可的边缘
            on_the_line.append([m-1, n-1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > low_boundary):
            img_yizhi[m - 1, n] = 255
            on_the_line.append([m - 1, n])
        if (a[0, 2] < high_boundary) and (a[0, 2] > low_boundary):
            img_yizhi[m - 1, n + 1] = 255
            on_the_line.append([m - 1, n + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > low_boundary):
            img_yizhi[m, n - 1] = 255
            on_the_line.append([m, n - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > low_boundary):
            img_yizhi[m, n + 1] = 255
            on_the_line.append([m, n + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > low_boundary):
            img_yizhi[m + 1, n - 1] = 255
            on_the_line.append([m + 1, n - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > low_boundary):
            img_yizhi[m + 1, n] = 255
            on_the_line.append([m + 1, n])
        if (a[2, 2] < high_boundary) and (a[2, 2] > low_boundary):
            img_yizhi[m + 1, n + 1] = 255
            on_the_line.append([m + 1, n + 1])
        # 做完双阈值检测后变成0/255二值图了
    # 再检查一遍
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()




















    

















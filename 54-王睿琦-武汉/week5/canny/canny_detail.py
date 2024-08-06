import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)  # 读取图片
    print("image", img)  # 打印图片数组
    if pic_path[-4:] == '.png':  # 判断图片格式是否为.png
        img = img * 255  # 如果是.png格式，扩展图片像素值到0-255范围
    img = img.mean(axis=-1)  # 对图片进行灰度化处理，取每个像素的均值

    # 1、高斯平滑
    sigma = 0.5  # 定义高斯平滑的标准差
    dim = 5  # 定义高斯核的尺寸
    Gaussian_filter = np.zeros([dim, dim])  # 创建一个dim x dim的高斯滤波器数组
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列，用于计算高斯核
    n1 = 1 / (2 * math.pi * sigma**2)  # 计算高斯核的常数部分
    n2 = -1 / (2 * sigma**2)  # 计算高斯核的指数部分
    for i in range(dim):  # 遍历高斯核的每一行
        for j in range(dim):  # 遍历高斯核的每一列
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i]**2 + tmp[j]**2))  # 计算高斯核的值
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()  # 归一化高斯核
    dx, dy = img.shape  # 获取图片的尺寸
    img_new = np.zeros(img.shape)  # 创建一个与原图像大小相同的数组用于存储平滑后的图像
    tmp = dim // 2  # 计算填充尺寸
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 对图像进行边缘填补
    for i in range(dx):  # 遍历图像的每一行
        for j in range(dy):  # 遍历图像的每一列
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)  # 进行高斯平滑
    plt.figure(1)  # 创建一个新的绘图窗口
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 显示平滑后的图像
    plt.axis('off')  # 关闭坐标轴

    # 2、求梯度。以下两个是滤波用的Sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 定义Sobel核用于x方向
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # 定义Sobel核用于y方向
    img_tidu_x = np.zeros(img_new.shape)  # 创建一个数组用于存储x方向的梯度图像
    img_tidu_y = np.zeros([dx, dy])  # 创建一个数组用于存储y方向的梯度图像
    img_tidu = np.zeros(img_new.shape)  # 创建一个数组用于存储梯度幅值图像
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 对平滑后的图像进行边缘填补
    for i in range(dx):  # 遍历图像的每一行
        for j in range(dy):  # 遍历图像的每一列
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # 计算x方向梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # 计算y方向梯度
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)  # 计算梯度幅值
    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 防止除以零
    angle = img_tidu_y / img_tidu_x  # 计算梯度角度
    plt.figure(2)  # 创建一个新的绘图窗口
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')  # 显示梯度幅值图像
    plt.axis('off')  # 关闭坐标轴

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)  # 创建一个数组用于存储非极大值抑制后的图像
    for i in range(1, dx - 1):  # 遍历图像的每一行（排除边缘）
        for j in range(1, dy - 1):  # 遍历图像的每一列（排除边缘）
            flag = True  # 标记是否要抹去
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 提取梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 判断梯度方向
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]  # 线性插值
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]  # 线性插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断是否是局部极大值
                    flag = False  # 如果不是，则抹去
            elif angle[i, j] >= 1:  # 判断梯度方向
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]  # 线性插值
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]  # 线性插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断是否是局部极大值
                    flag = False  # 如果不是，则抹去
            elif angle[i, j] > 0:  # 判断梯度方向
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]  # 线性插值
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]  # 线性插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断是否是局部极大值
                    flag = False  # 如果不是，则抹去
            elif angle[i, j] < 0:  # 判断梯度方向
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]  # 线性插值
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]  # 线性插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):  # 判断是否是局部极大值
                    flag = False  # 如果不是，则抹去
            if flag:  # 如果是局部极大值
                img_yizhi[i, j] = img_tidu[i, j]  # 保留该像素值
    plt.figure(3)  # 创建一个新的绘图窗口
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示非极大值抑制后的图像
    plt.axis('off')  # 关闭坐标轴

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点，查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5  # 计算低阈值
    high_boundary = lower_boundary * 3  # 计算高阈值，这里设置高阈值是低阈值的三倍
    zhan = []  # 创建一个空列表作为栈
    for i in range(1, img_yizhi.shape[0] - 1):  # 遍历图像的每一行（排除边缘）
        for j in range(1, img_yizhi.shape[1] - 1):  # 遍历图像的每一列（排除边缘）
            if img_yizhi[i, j] >= high_boundary:  # 如果当前像素值大于高阈值
                img_yizhi[i, j] = 255  # 标记为边缘
                zhan.append([i, j])  # 将坐标进栈
            elif img_yizhi[i, j] <= lower_boundary:  # 如果当前像素值小于低阈值
                img_yizhi[i, j] = 0  # 标记为非边缘

    while not len(zhan) == 0:  # 当栈不为空时
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]  # 提取8邻域
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1 - 1, temp_2] = 255  # 标记为边缘
            zhan.append([temp_1 - 1, temp_2])  # 进栈
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255  # 标记为边缘
            zhan.append([temp_1 - 1, temp_2 + 1])  # 进栈
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1, temp_2 - 1] = 255  # 标记为边缘
            zhan.append([temp_1, temp_2 - 1])  # 进栈
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1, temp_2 + 1] = 255  # 标记为边缘
            zhan.append([temp_1, temp_2 + 1])  # 进栈
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255  # 标记为边缘
            zhan.append([temp_1 + 1, temp_2 - 1])  # 进栈
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1 + 1, temp_2] = 255  # 标记为边缘
            zhan.append([temp_1 + 1, temp_2])  # 进栈
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):  # 判断8邻域内是否有可能是边缘的点
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255  # 标记为边缘
            zhan.append([temp_1 + 1, temp_2 + 1])  # 进栈

    for i in range(img_yizhi.shape[0]):  # 遍历图像的每一行
        for j in range(img_yizhi.shape[1]):  # 遍历图像的每一列
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:  # 如果当前像素值不等于0或255
                img_yizhi[i, j] = 0  # 标记为非边缘

    # 绘图
    plt.figure(4)  # 创建一个新的绘图窗口
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示双阈值检测后的图像
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示绘图窗口

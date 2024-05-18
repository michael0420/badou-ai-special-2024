import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    print("image",img)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化

    # 1、高斯平滑
    dim = 5
    sigma = 1.0

    # 创建一个大小为 dim x dim 的零矩阵（Gaussian_filter），用于存储高斯核的值
    Gaussian_filter = np.zeros((dim, dim))

    # 生成一个列表（tmp），包含了从 -dim//2 到 dim//2 的整数
    tmp = [i - dim // 2 for i in range(dim)]

    # 计算高斯函数的两个常数部分（n1 和 n2）
    n1 = 1 / (2 * np.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    # 计算高斯函数的值
    for i in range(dim):
        for j in range(dim):
            # 计算高斯函数的指数部分
            Gaussian_filter[i, j] = n1 * np.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    # 归一化
    Gaussian_filter /= Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros([dx, dy])
    img_pad = np.pad(img, ((dim//2, dim//2), (dim//2, dim//2)), 'constant')  # dim//2是向下取整
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    print("img_new",img_new)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray') # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    plt.show()

    # 2、求梯度。以下两个是滤波用的scharr算子
    # 定义 Scharr 算子，这些值的具体选择（例如，为什么是3和10，而不是其他的值）是为了使 Scharr 算子在所有方向上都有相同的能量响应，这样可以提供更好的旋转一致性。
    scharr_kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    scharr_kernel_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])

    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度x方向
    img_tidu_y = np.zeros([dx, dy])  # 存储梯度y方向
    img_tidu = np.zeros(img_new.shape) # 存储梯度幅值
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * scharr_kernel_x) # padding后的图像与算子卷积
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * scharr_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2) # 平方和开根号

    img_tidu_x[img_tidu_x == 0] = 0.00000001  # 防止分母为0
    angle = img_tidu_y / img_tidu_x #

    print("tidu",img_tidu)
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    # 3、非极大值抑制
    def calculate_nums(temp, angle_i_j):
        # 根据角度值计算 num_1 和 num_2
        if angle_i_j <= -1:
            # 当角度小于等于-1时，使用左上角和右下角的点进行计算
            return (temp[0, 1] - temp[0, 0]) / angle_i_j + temp[0, 1], (temp[2, 1] - temp[2, 2]) / angle_i_j + temp[
                2, 1]
        elif angle_i_j >= 1:
            # 当角度大于等于1时，使用右上角和左下角的点进行计算
            return (temp[0, 2] - temp[0, 1]) / angle_i_j + temp[0, 1], (temp[2, 0] - temp[2, 1]) / angle_i_j + temp[
                2, 1]
        elif angle_i_j > 0:
            # 当角度大于0时，使用右上角和左下角的点进行计算
            return (temp[0, 2] - temp[1, 2]) * angle_i_j + temp[1, 2], (temp[2, 0] - temp[1, 0]) * angle_i_j + temp[
                1, 0]
        elif angle_i_j < 0:
            # 当角度小于0时，使用左上角和右下角的点进行计算
            return (temp[1, 0] - temp[0, 0]) * angle_i_j + temp[1, 0], (temp[1, 2] - temp[2, 2]) * angle_i_j + temp[
                1, 2]


    img_yizhi = np.zeros(img_tidu.shape)  # 创建一个与 img_tidu 形状相同的零矩阵，用于存储非极大值抑制的结果
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 获取当前像素的 8 邻域
            num_1, num_2 = calculate_nums(temp, angle[i, j])  # 计算 num_1 和 num_2
            # 如果当前像素的梯度值大于 num_1 和 num_2，则认为它是极大值，将其梯度值保存到结果中
            if img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2:
                img_yizhi[i, j] = img_tidu[i, j]
    print("yizhi",img_yizhi)
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进队列
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 高阈值是低阈值的三倍
    queue = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 如果像素值大于高阈值，标记为边缘
                img_yizhi[i, j] = 255
                queue.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 如果像素值小于低阈值，标记为非边缘
                img_yizhi[i, j] = 0

    while queue:  # 当队列为空时结束
        temp_1, temp_2 = queue.pop(0)  # 出队列
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:  # 8邻域
            nx, ny = temp_1 + dx, temp_2 + dy
            if (a[dx + 1, dy + 1] < high_boundary) and (a[dx + 1, dy + 1] > lower_boundary): # 如果像素值在两个阈值之间
                img_yizhi[nx, ny] = 255  # 这个像素点标记为边缘
                queue.append([nx, ny])  # 进队列

    for i in range(img_yizhi.shape[0]): # 将非边缘点置为0
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255: # 弱边缘点改为0
                img_yizhi[i, j] = 0
    newyizhi = img_yizhi
    # 绘图
    print("newyizhi", newyizhi)
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
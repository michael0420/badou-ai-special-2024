import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
if __name__ == '__main__':
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    # print("image", img)
    if pic_path[-4:] == '.png':  # plt读取png图片是0到1的浮点数，所以要扩展到255再计算（cv2不是）
        img = img*255
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """
    axis=-1 指定了沿着最后一个轴（在这种情况下是颜色通道轴）计算平均值。
    对于 RGB 图像，这意味着对每个像素的三个颜色通道（红、绿、蓝）取平均值，从而得到一个灰度值。
    这是将彩色图像转换为灰度图像的常用方法。
    """
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化

    # 1、高斯平滑
    sigma = 1
    dim = 5  # 高斯卷积核大小 trade off
    Gaussian_filter = np.zeros([dim, dim])  # 创建5*5的二维数组
    tmp = [i-dim//2 for i in range(dim)]  # 计算高斯核中每个位置相对于核中心的偏移量
    # print(tmp)
    n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核（左半部分）
    n2 = -1/(2*sigma**2)  # 指数部分
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    pad = dim//2
    """
    np.pad 函数的第一个参数是要填补的数组（在这里是 img），第二个参数是一个由元组组成的列表，指定了每个维度上的填补尺寸。
    在这个例子中，有两个维度（通常对应于图像的高度和宽度），因此有两个元组：((pad, pad), (pad, pad))。
第一个元组 (pad, pad) 指定了第一个维度（高度）上的填补尺寸。在图像上下两侧各填补 pad 个像素。
第二个元组 (pad, pad) 指定了第二个维度（宽度）上的填补尺寸。在图像左右两侧各填补 pad 个像素。
'constant' 是 np.pad 函数的 mode 参数，它指定了填补的方式。'constant' 意味着使用一个常数值来填补，但默认情况下，这个常数值是0。
    """
    img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'constant')
    # 如果在没有填补的情况下直接应用滤波器，那么图像的边界像素将只受到滤波器的一部分影响，这可能导致边界处的处理结果不准确或不一致
    # print(img_pad)
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')  # 用于关闭或隐藏当前坐标轴（即 x 轴和 y 轴）的显示

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros(img_new.shape)
    img_tidu = np.zeros(img_new.shape)
    img_pad1 = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 3*3滤波器（3//2=1）
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad1[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad1[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2+img_tidu_y[i, j]**2)  # np.sqrt求平方根
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    tan_angle = img_tidu_y/img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap="gray")
    plt.axis('off')

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):  # 这样循环实际上是取去除了图像边缘一圈的部分，因为没有加padding
            flag = True  # 在8邻域内是否要保留做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            if tan_angle[i, j] <= -1:  # pi/2-3/4pi
                num_1 = (temp[0, 1] - temp[0, 0]) / tan_angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan_angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan_angle[i, j] >= 1:  # pi/4-pi/2
                num_1 = (temp[0, 2] - temp[0, 1]) / tan_angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan_angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan_angle[i, j] > 0:  # 0-pi/4
                num_1 = (temp[0, 2] - temp[1, 2]) * tan_angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan_angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan_angle[i, j] < 0:  # 3/4pi-pi
                num_1 = (temp[1, 0] - temp[0, 0]) * tan_angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan_angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean()*0.5  # 随便设的，不同阈值会有不同效果
    higher_boundary = lower_boundary*3
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= higher_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])  # 用栈来存储强边缘像素
            elif img_yizhi[i, j] <= lower_boundary:
                img_yizhi[i, j] = 0  # 舍
    while not len(zhan) == 0:  # while not 后面跟着一个条件表达式，只要这个条件表达式的值为 False，循环就会继续执行
        temp_1, temp_2 = zhan.pop()
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]  # 遍历所有强边缘像素点，因为弱边缘像素点八邻域内一定有强边缘，反推
        if (a[0, 0] < higher_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255
            zhan.append([temp_1 - 1, temp_2 - 1])
        if (a[0, 1] < higher_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < higher_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < higher_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < higher_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < higher_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < higher_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < higher_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()

"""
关于if和elif的使用，elif是“else if”的缩写，它用于检查多个条件，但只在第一个为真的条件处执行相应的代码块。
使用两个独立的if语句来确保两个条件都被检查.
"""
























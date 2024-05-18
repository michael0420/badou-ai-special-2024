import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == "__main__":
    pic_path = "lenna.png"
    img = plt.imread(pic_path)
    print("样本集img\n",img)
    if pic_path[-4:] == ".png":     # [-4:]   切片，截取倒数第4位至最后一位
        # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255             # 还是浮点数类型
    img = img.mean(axis = -1)       # 取均值的方法进行灰度化
    print("样本集img灰度化\n", img)

    # 1.高斯平滑
    # sigma = 1.52   高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5
    dim = 5      # 高斯核尺寸
    Gaussion_filter = np.zeros([dim,dim])   # 储存高斯核，这是数组，不是列表
    tmp = [i - dim//2 for i in range(dim)]  # 生成一个序列
    print("tmp:",tmp)
    # 计算高斯核，公式：n1为e左侧数据，n2为e的指数数据。math.pi:圆周率π
    n1 = 1/(2 * math.pi * sigma**2)
    n2 = -1/(2 * sigma**2)
    for i in range(dim):
        for j in range(dim):          # math.exp:e的幂次方
            Gaussion_filter[i,j] = n1 * math.exp(n2*(tmp[i]**2 + tmp[j]*22))
    # 1）归一化后加快了梯度下降求最优解的速度；2）归一化有可能提高精度（如KNN）
    Gaussion_filter = Gaussion_filter / Gaussion_filter.sum()    # 归一化

    dx,dy = img.shape      # 获取原图的行列数
    print("dx:",dx,"\ndy:",dy)
    img_new = np.zeros(img.shape)   # 储存平滑后的图像，zeros函数得到的是浮点型数据
    """ np.pad()图像边缘填充技术，
    即在图像四周边缘填充0，使得卷积运算后图像大小不会缩小，同时也不会丢失边缘和角落的信息
    在卷积神经网络中，通常采用constant填充方式"""
    r = dim//2  # 高斯核半径为2
    img_pad = np.pad(img,((r,r),(r,r)),"constant")
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim] * Gaussion_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8),cmap="gray")    # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis("off")
    # plt.show()

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)
    img_tidu_y = np.zeros([dx,dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new,((1,1),(1,1)),"constant")
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)  # X方向
            img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)  # Y方向
            img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2)
    img_tidu_x[img_tidu_x == 0] = 0.000000001
    tan = img_tidu_y/img_tidu_x
    print("tan:\n",tan)
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8),cmap="gray")
    plt.axis("off")
    # plt.show()

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    print("img_yizhi:\n",img_yizhi)
    for i in range(1,dx - 1):       # 没有做np.pad 所以边缘像素点要去掉
        for j in range(1,dy - 1):   # 没有做np.pad 所以边缘像素点要去掉
            flag = True             # 在8个邻域内是否要抹去做个标记
            temp = img_tidu[i-1:i+2, j-1:j+2]   # 梯度幅值的8邻域矩阵
            if tan[i,j] <= -1:
                num_1 = (temp[0,1] - temp[0,0])/tan[i,j] + temp[0,1]
                num_2 = (temp[2,1] - temp[2,2])/tan[i,j] + temp[2,1]
                if not (img_tidu[i,j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan[i,j] >= 1:
                num_1 = (temp[0,2] - temp[0,1])/tan[i,j] + temp[0,1]
                num_2 = (temp[2,0] - temp[2,1])/tan[i,j] + temp[2,1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan[i,j] > 0:
                num_1 = (temp[0,2] - temp[1,2])/tan[i,j] + temp[1,2]
                num_2 = (temp[2,0] - temp[1,0])/tan[i,j] + temp[1,0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif tan[i,j]:
                num_1 = (temp[1,0] - temp[0,0])/tan[i,j] + temp[1,0]
                num_2 = (temp[1,2] - temp[2,2])/tan[i,j] + temp[1,2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i,j] = img_tidu[i,j]
    print("img_yizhi后:\n", img_yizhi)
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8),cmap="gray")
    plt.axis("off")
    # plt.show()

    # 4、双阈值检测，链接边缘。遍历所有一定是边的点，查看8邻域是否存在可能是边的点
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1,img_yizhi.shape[0]-1):     # 外圈不考虑了
        for j in range(1,img_yizhi.shape[1]-1):
            if img_yizhi[i,j] >= high_boundary:     # 大于高阈值为强边缘，留
                img_yizhi[i,j] = 255
                zhan.append([i,j])
            elif img_yizhi[i,j] <= lower_boundary:  # 小于低阈值不是边缘
                img_yizhi[i,j] = 0

    while not len(zhan) == 0:
        temp_1,temp_2 = zhan.pop()
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        print("a:\n",a)
        """
        小于高阈值并且大于低阈值为弱边缘，弱化边缘周围8邻域有强边缘，则保留为真是边缘，
        逆向思考，强边缘的8邻域周围有弱边缘，则保留该弱边缘，记录为强边缘
        """
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

    # 经过强边缘检测完毕后，要么为0，要么为255。还有不在强边缘周边的点，均定义为非边缘。
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i,j] != 0 and img_yizhi[i,j] != 255:
                img_yizhi[i,j] = 0
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()

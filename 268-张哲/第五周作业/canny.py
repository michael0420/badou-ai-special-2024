import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
def canny_detail(img,sigma,dim,):
    #1.高斯平滑
    Gaussian_filter = np.zeros([dim,dim])# 存储高斯核，这是数组
    tmp = [i-dim//2 for i in range(dim)]# 生成一个序列
    # 计算高斯核
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter/Gaussian_filter.sum()
    x,y = img.shape
    img_f= np.zeros(img.shape) # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim//2
    img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')#边缘填补
    for i in range(x):
        for j in range(y):
            img_f[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gaussian_filter)
    plt.figure(1)
    plt.imshow(img_f.astype(np.uint8), cmap='gray')  # 此时的img_f是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    
    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros([x, y])  # 存储梯度图像
    img_tidu_y = np.zeros([x, y])
    img_tidu = np.zeros(img_f.shape)
    img_pad = np.pad(img_f, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(x):
        for j in range(y):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3、非极大值抑制
    img_nms = np.zeros(img_tidu.shape)
    for i in range(1, x - 1):
        for j in range(1, y - 1):
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
                img_nms[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_nms.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_nms.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_nms.shape[1] - 1):
            if img_nms[i, j] >= high_boundary:  # 取，一定是边的点
                img_nms[i, j] = 255
                zhan.append([i, j])
            elif img_nms[i, j] <= lower_boundary:  # 舍
                img_nms[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_nms[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_nms[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_nms[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_nms[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_nms[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_nms[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_nms[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_nms[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_nms[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_nms.shape[0]):
        for j in range(img_nms.shape[1]):
            if img_nms[i, j] != 0 and img_nms[i, j] != 255:
                img_nms[i, j] = 0

    # 绘图
    plt.figure(4)
    plt.imshow(img_nms.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()


if __name__ == '__main__':
    img_path = 'lenna.png'
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差
    dim = 5  # 高斯核尺寸
    canny_detail(gray,sigma,dim)
#   快捷方式
    img_canny = cv2.Canny(gray, 200, 300)
    cv2.imshow("canny",img_canny)
    cv2.waitKey()
    cv2.destroyAllWindows()
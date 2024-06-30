import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
def canny_existed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    can = cv2.Canny(img, 200, 300)
    return can

def canny_pricinple(img):
    print("image", img)
    if img.dtype == np.float32:  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化

    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    dim = 5  # 高斯核尺寸
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    # plt.figure(1)
    # plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    # plt.axis('off')

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    # plt.figure(2)
    # plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    # plt.axis('off')

    # 3、非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
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
                img_yizhi[i, j] = img_tidu[i, j]
    # plt.figure(3)
    # plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    # plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
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

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    return img_yizhi


def CannyThreshold(lowThreshold):
    # detected_edges = cv2.GaussianBlur(gray,(3,3),0) #高斯滤波
    detected_edges = cv2.Canny(img_gray,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize=kernel_size)  # 边缘检测

    # 用原始颜色添加到检测的边缘上。
    # 按位“与”操作。对于每个像素,将两幅输入图像相应位置的像素值分别进行按位“与”运算,输出的结果图像的对应像素值即为这两幅输入图像对应像素值的按位与结果。
    # src1和src2表示要进行按位“与”操作的两幅输入图像；
    # mask 是可选参数，如果指定了掩膜，则只对掩膜对应位置的像素进行按位“与”操作。函数的返回值表示按位“与”运算的结果。
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    cv2.imshow('canny result', dst)

if __name__ == '__main__':
    lowThreshold = 0
    max_lowThreshold = 400
    ratio = 3
    kernel_size = 3

    img = cv2.imread('/Original_Data/0216.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图

    # cv2.namedWindow('canny result')

    # # 设置调节杠,
    # '''
    # 下面是第二个函数，cv2.createTrackbar()
    # 共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
    # 第一个参数，是这个trackbar对象的名字
    # 第二个参数，是这个trackbar对象所在面板的名字
    # 第三个参数，是这个trackbar的默认值,也是调节的对象
    # 第四个参数，是这个trackbar上调节的范围(0~count)
    # 第五个参数，是调节trackbar时调用的回调函数名
    # '''
    # cv2.createTrackbar('Min threshold', 'canny result', lowThreshold, max_lowThreshold, CannyThreshold)
    #
    # CannyThreshold(0)  # initialization
    # if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    #     cv2.destroyAllWindows()

    '''
    Sobel算子
    Sobel算子函数原型如下：
    dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
    前四个是必须的参数：
    第一个参数是需要处理的图像；
    第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
    dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
    其后是可选的参数：
    dst是目标图像；
    ksize是Sobel算子的大小，必须为1、3、5、7。
    scale是缩放导数的比例常数，默认情况下没有伸缩系数；
    delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
    borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
    '''

    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # 对x求导
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # 对y求导

    # Laplace 算子
    img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)

    # Canny 算子
    img_canny = cv2.Canny(img_gray, 100, 150)

    plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")
    plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")
    plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")
    plt.subplot(234), plt.imshow(img_laplace, "gray"), plt.title("Laplace")
    plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
    plt.show()


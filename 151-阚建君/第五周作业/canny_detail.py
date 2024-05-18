import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == '__main__':
    '''
        算法步骤：
            1.灰度化
            2.高斯平滑降噪
            3.求梯度
            4.非极大值抑制
            5.双阈值 高阈值强边缘 低阈值不是阈值  介于之间弱边缘（对于弱边缘需要进行抑制孤立低阈值点）
    '''
    # 灰度化
    pic_path = '../lenna.png'
    img = plt.imread(pic_path)
    print("image", img)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    plt.figure(1)
    plt.imshow(img,cmap='gray')
    print(img)
    # 高斯平滑
    sigma = 0.2
    kernel = 4
    Gaussian_filter = np.zeros([kernel, kernel]) # 数组对象
    tmp = [i - kernel // 2 for i in range(kernel)]
    n1 = 1 / (2 * math.pi * sigma ** 2)  #  高斯卷积核左部竖式
    n2 = -1 / (2 * sigma ** 2)  #  高斯卷积核右部竖式
    for i in range(kernel):
        for j in range(kernel):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))   # 套用公式，计算高斯核矩阵组成
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()    # 归一化处理
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    tmp = kernel // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + kernel, j:j + kernel] * Gaussian_filter)  #
    plt.figure(2)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 强制类型转换，转成灰度图
    plt.axis('off')

    # 求梯度 检测图像中的水平、垂直、对角边缘
    # 函数法
    # x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    # y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    #
    # absX = cv2.convertScaleAbs(x)
    # absy = cv2.convertScaleAbs(y)
    #
    # dst = cv2.addWeighted(absX, 0.5, absy, 0.5, 0)
    # plt.figure(3)
    # plt.imshow(dst.astype(np.uint8), cmap='gray')
    # plt.axis('off')

    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])    # 赋值矩阵用
    img_tidu = np.zeros(img_new.shape)  # 初始化梯度
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向  +3是根据sobel核的维度进行确定的
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)   # 根号下 x2+y2 该点梯度
    img_tidu_x[img_tidu_x == 0] = 0.00000001   #因为求tan  y/x 分母不为0    不影响当前为0的结果大小
    angle = img_tidu_y/img_tidu_x
    plt.figure(3)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape) # 初始化
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = False   # 如果当前像素梯度比其他两个点像素大 则保留  否则直接抑制置0   是否保留该像素梯度
            temp = img_tidu[i-1:i+2,j-1:j+2] # 构造8邻域矩阵
            # 角度判断所在象限
            if angle[i,j]<=-1:
                # 使用线性插值判断是否抑制   tan = y/x
                num_1 = (temp[0,1]-temp[0,0])/angle[i,j] + temp[0,1]    #  最后加上的是角点的值 计算出实际的值
                num_2 = (temp[2,1]-temp[2,2])/angle[i,j] + temp[2,1]
                if img_tidu[i,j]>num_1 and img_tidu[i,j]>num_2:
                    flag = True # 保留
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2:
                    flag = True # 保留
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2:
                    flag = True # 保留
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2:
                    flag = True # 保留
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里设置高阈值是低阈值的三倍
    stack = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 强边缘
                img_yizhi[i, j] = 255
                stack.append([i, j])    # 进栈
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(stack) == 0:
        temp_1, temp_2 = stack.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]  # 构造八邻域
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):   # 判断8邻域值是否抑制  是否有连接了强边缘
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            stack.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            stack.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            stack.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            stack.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            stack.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            stack.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            stack.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            stack.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:     # 边缘值界限判定
                img_yizhi[i, j] = 0

    # 绘图
    plt.figure(5)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()



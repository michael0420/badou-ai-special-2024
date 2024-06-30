import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

if __name__ == "__main__":
    img_path = 'lenna.png'
    plt_imread = plt.imread("lenna.png")
    print('picture', plt_imread)
    if img_path[-4:] == '.png':
        plt_imread = plt_imread * 255
    # 灰度化
    plt_imread = plt_imread.mean(axis=-1)
    # 高斯降噪处理
    # μ高斯核尺寸
    # 其标准差σ决定了分布的幅度（胖瘦）
    σ = 0.618
    μ = 5
    gaosijuanjihe = np.zeros([μ, μ]) # 初始化卷积核
    # temp = np.linspace(-2, 5, 1)
    temp = [i - μ // 2 for i in range(μ)]  # 生成一个序列
    # 带入高斯平滑公式
    n1 = 1 / (2 * math.pi * σ ** 2)
    n2 = -1 / (2 * σ ** 2)
    for i in range(μ):
        for j in range(μ):
            gaosijuanjihe[i, j] = n1 * math.exp(n2 * (temp[i] ** 2 + temp[j] ** 2))

    gaosijuanjihe = gaosijuanjihe / gaosijuanjihe.sum()
    dx, dy = plt_imread.shape
    # 存储高斯卷积之后的结果
    zeros_img = np.zeros(plt_imread.shape)
    temp = μ // 2
    img_pad = np.pad(plt_imread, ((temp, temp), (temp, temp)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            # 进行加权求和
            zeros_img[i, j] = np.sum(img_pad[i:i + μ, j:j + μ] * gaosijuanjihe)
    plt.figure(1)
    plt.imshow(zeros_img.astype(np.uint8), cmap='gray')  # 此时的zeros_img是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 第三步进行sobe边缘检测
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])  # x方向的卷积核
    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])  # y方向的卷积核
    # 与原图大小一样
    tan_x = np.zeros(zeros_img.shape)
    # 与原图大小一样
    tan_y = np.zeros([dx, dy])
    # 与原图大小一样保存做过sobe边缘检查后的图像
    tan_new = np.zeros(zeros_img.shape)
    tan_pad = np.pad(zeros_img, ((1, 1), (1, 1)), "constant")
    for i in range(dx):
        for j in range(dy):
            tan_x[i, j] = np.sum(tan_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向做卷积
            tan_y[i, j] = np.sum(tan_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向做卷积
            # sqrt是根号的意思
            tan_new[i, j] = np.sqrt(tan_x[i, j] ** 2 + tan_y[i, j] ** 2)
    tan_x[tan_x == 0] = 0.00000001
    ti_du = tan_y / tan_x
    plt.figure(2)
    plt.imshow(tan_new.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # 第四步进行非极大值抑制, 做减法，保留最优边缘
    zuiyou_bianyuan = np.zeros(tan_new.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            ba_juzheng = tan_new[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if ti_du[i, j] <= -1:  # 第四象限
                temp_1 = (ba_juzheng[0, 1] - ba_juzheng[0, 0]) / ti_du[i, j] + ba_juzheng[0, 1]
                temp_2 = (ba_juzheng[2, 1] - ba_juzheng[2, 2]) / ti_du[i, j] + ba_juzheng[2, 1]
                if not (tan_new[i, j] > temp_1 and tan_new[i, j] > temp_2):
                    flag = False
            elif ti_du[i, j] >= 1:
                temp_1 = (ba_juzheng[0, 2] - ba_juzheng[0, 1]) / ti_du[i, j] + ba_juzheng[0, 1]
                temp_2 = (ba_juzheng[2, 0] - ba_juzheng[2, 1]) / ti_du[i, j] + ba_juzheng[2, 1]
                if not (tan_new[i, j] > temp_1 and tan_new[i, j] > temp_2):
                    flag = False
            elif ti_du[i, j] > 0:
                temp_1 = (ba_juzheng[0, 2] - ba_juzheng[1, 2]) * ti_du[i, j] + ba_juzheng[1, 2]
                temp_2 = (ba_juzheng[2, 0] - ba_juzheng[1, 0]) * ti_du[i, j] + ba_juzheng[1, 0]
                if not (tan_new[i, j] > temp_1 and tan_new[i, j] > temp_2):
                    flag = False
            elif ti_du[i, j] < 0:
                temp_1 = (ba_juzheng[1, 0] - ba_juzheng[0, 0]) * ti_du[i, j] + ba_juzheng[1, 0]
                temp_2 = (ba_juzheng[1, 2] - ba_juzheng[2, 2]) * ti_du[i, j] + ba_juzheng[1, 2]
                if not (tan_new[i, j] > temp_1 and tan_new[i, j] > temp_2):
                    flag = False
            if flag:
                zuiyou_bianyuan[i, j] = tan_new[i, j]
    plt.figure(3)
    plt.imshow(zuiyou_bianyuan.astype(np.uint8), cmap='gray')
    plt.axis('off')
    # 第五步进行双阈值检测
    di_yuzhi = tan_new.mean() * 0.5
    gao_yuzhi = di_yuzhi * 3
    link_list = []
    for i in range(1, zuiyou_bianyuan.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, zuiyou_bianyuan.shape[1] - 1):
            if zuiyou_bianyuan[i, j] >= gao_yuzhi:
                zuiyou_bianyuan[i, j] = 255  # 白颜色 保留
                link_list.append([i, j])
            elif zuiyou_bianyuan[i, j] <= di_yuzhi:
                zuiyou_bianyuan[i, j] = 0  # 纯黑 舍弃

    # 增强对比度
    while not len(link_list) == 0:
        temp_1, temp_2 = link_list.pop()  # 出栈
        a = zuiyou_bianyuan[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2] # 八邻域
        if (a[0, 0] < gao_yuzhi) and (a[0, 0] > di_yuzhi):  # 保留为边缘点
            zuiyou_bianyuan[temp_1 - 1, temp_2 - 1] = 255
            link_list.append([temp_1-1, temp_2-1])  # 进栈
        if (a[0, 1] < gao_yuzhi) and (a[0, 1] > di_yuzhi):
            zuiyou_bianyuan[temp_1 - 1, temp_2] = 255
            link_list.append([temp_1 - 1, temp_2])
        if (a[0, 2] < gao_yuzhi) and (a[0, 2] > di_yuzhi):
            zuiyou_bianyuan[temp_1 - 1, temp_2 + 1] = 255
            link_list.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < gao_yuzhi) and (a[1, 0] > di_yuzhi):
            zuiyou_bianyuan[temp_1, temp_2 - 1] = 255
            link_list.append([temp_1, temp_2 - 1])
        if (a[1, 2] < gao_yuzhi) and (a[1, 2] > di_yuzhi):
            zuiyou_bianyuan[temp_1, temp_2 + 1] = 255
            link_list.append([temp_1, temp_2 + 1])
        if (a[2, 0] < gao_yuzhi) and (a[2, 0] > di_yuzhi):
            zuiyou_bianyuan[temp_1 + 1, temp_2 - 1] = 255
            link_list.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < gao_yuzhi) and (a[2, 1] > di_yuzhi):
            zuiyou_bianyuan[temp_1 + 1, temp_2] = 255
            link_list.append([temp_1 + 1, temp_2])
        if (a[2, 2] < gao_yuzhi) and (a[2, 2] > di_yuzhi):
            zuiyou_bianyuan[temp_1 + 1, temp_2 + 1] = 255
            link_list.append([temp_1 + 1, temp_2 + 1])

    for i in range(zuiyou_bianyuan.shape[0]):
        for j in range(zuiyou_bianyuan.shape[1]):
            if zuiyou_bianyuan[i, j] != 0 and zuiyou_bianyuan[i, j] != 255:
                zuiyou_bianyuan[i, j] = 0  # 舍弃

        # 绘图
    plt.figure(4)
    plt.imshow(zuiyou_bianyuan.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
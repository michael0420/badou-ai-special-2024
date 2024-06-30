# coding = utf-8

'''
        实现canny算法
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

'''
使用接口实现canny
'''
# origin_img = cv2.imread('lenna.png')
# cv2.imshow('origin_img', origin_img)
#
# gray_1 = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(gray_1, 100, 300)       # 参数1：灰度图；参数2：低阈值；参数3：高阈值
# cv2.imshow('canny', canny)

'''
详细实现canny算法
'''
if __name__ == '__main__':
    #   一：读图与灰度化
    img_path = 'lenna.png'
    ori = cv2.imread(img_path)
    cv2.imshow('ori', ori)
    plt.subplot(231)
    plt.imshow(ori)
    if img_path[-4:] == '.png':     # .png格式读取后为0-1的浮点数
        ori = ori * 255
    gray_2 = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)      # 灰度化
    cv2.imshow('gray', gray_2)
    plt.subplot(232)
    plt.imshow(gray_2)

    #   二：高斯平滑
    sigma = 1.5     # 设置高斯核参数，标准差
    dim = 5         # 设置高斯核尺寸,一般为5*5
    Gauss_filter = np.zeros([dim, dim])   # 创建空矩阵

    tmp1 = [i - dim // 2 for i in range(dim)]
    # i-dim//2 对于序列中的每个整数i，计算它相对于dim//2（即核尺寸一半，向下取整）的偏移量。在5x5的核中，dim//2等于2，所以偏移量序列是[-2, -1, 0, 1, 2]
    g1 = 1 / (2 * math.pi * sigma**2)       # 代入高斯核公式计算前置因子g1
    g2 = -1 / (2 * sigma**2)                # 代入高斯核公式计算前置因子g2
    # 双层循环计算高斯核
    # 计算其相对于中心的偏移量的平方和，然后存储该位置的高斯值
    for i in range(dim):
        for j in range(dim):
            Gauss_filter[i, j] = g1 * math.exp(g2 * (tmp1[i]**2 + tmp1[j]**2))    # 代入公式，math.exp()计算指数项
    Gauss_filter = Gauss_filter / Gauss_filter.sum()        # 对高斯核进行归一化，使得其所有元素的和等于1，确保在应用滤波器时，图像的总体亮度不会改变

    gx, gy = gray_2.shape
    Gauss_img = np.zeros([gx, gy])    # 存储高斯平滑后的图像

    tmp2 = dim // 2
    img_pad_1 = np.pad(gray_2, ((tmp2, tmp2), (tmp2, tmp2)), 'constant')
    # 使用np.pad函数对原始图像img进行边缘填充,填充的大小是((tmp, tmp), (tmp, tmp))，这意味着在图像的顶部和底部各填充tmp行，在图像的左侧和右侧各填充tmp列
    # 填充的方式是'constant'，即使用常数（默认为0）进行填充
    # 进行卷积
    for i in range(gx):
        for j in range(gy):
            Gauss_img[i, j] = np.sum(img_pad_1[i:i+dim, j:j+dim] * Gauss_filter)
    cv2.imshow('Gauss', Gauss_img.astype(np.uint8))
    plt.subplot(233)
    plt.imshow(Gauss_img)

    #   三：全量边缘检测（可选择使用sobel、prewitt、laplace等算法）
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    # 创建空矩阵存储x，y，综合方向的sobel检测图片
    sobel_x_img = np.zeros(Gauss_img.shape)
    sobel_y_img = np.zeros([gx, gy])
    sobel_img = np.zeros(Gauss_img.shape)
    img_pad_2 = np.pad(Gauss_img, ((1, 1), (1, 1)), 'constant')     # 使用np.pad函数对原始图像img进行边缘填充
    for i in range(gx):
        for j in range(gy):
            sobel_x_img[i, j] = np.sum(img_pad_2[i:i+3, j:j+3] * sobel_x)
            sobel_y_img[i, j] = np.sum(img_pad_2[i:i+3, j:j+3] * sobel_y)
            sobel_img[i, j] = np.sqrt(sobel_x_img[i, j]**2 + sobel_y_img[i, j]**2)
    sobel_x_img[sobel_x_img == 0] = 0.00000001
    tan = sobel_y_img / sobel_x_img
    cv2.imshow('sobel', sobel_img.astype(np.uint8))
    plt.subplot(234)
    plt.imshow(sobel_img)

    #   四：非极大值抑制
    nms_img = np.zeros(sobel_img.shape)
    for i in range(1, gx - 1):
        for j in range(1, gy - 1):
            flag = True     # 初始标记为True，默认保留
            tmp3 = sobel_img[i-1:i+2, j-1:j+2]      # 在选取点的8个邻域范围内 [0,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[2,2]
            # 使用线性插值法判断是否要抑制
            if tan[i, j] <= -1:
                n1 = (tmp3[0, 1] - tmp3[0, 0]) / tan[i, j] + tmp3[0, 1]
                n2 = (tmp3[2, 1] - tmp3[2, 2]) / tan[i, j] + tmp3[2, 1]
                if not (sobel_img[i, j] > n1 and sobel_img[i, j] > n2):
                    flag = False
            elif tan[i, j] >= 1:
                n1 = (tmp3[0, 2] - tmp3[0, 1]) / tan[i, j] + tmp3[0, 1]
                n2 = (tmp3[2, 0] - tmp3[2, 1]) / tan[i, j] + tmp3[2, 1]
                if not (sobel_img[i, j] > n1 and sobel_img[i, j] > n2):
                    flag = False
            elif tan[i, j] > 0:
                n1 = (tmp3[0, 2] - tmp3[1, 2]) * tan[i, j] + tmp3[1, 2]
                n2 = (tmp3[2, 0] - tmp3[1, 0]) * tan[i, j] + tmp3[1, 0]
                if not (sobel_img[i, j] > n1 and sobel_img[i, j] > n2):
                    flag = False
            elif tan[i, j] < 0:
                n1 = (tmp3[1, 0] - tmp3[0, 0]) * tan[i, j] + tmp3[1, 0]
                n2 = (tmp3[1, 2] - tmp3[2, 2]) * tan[i, j] + tmp3[1, 2]
                if not (sobel_img[i, j] > n1 and sobel_img[i, j] > n2):
                    flag = False
            if flag:
                nms_img[i, j] = sobel_img[i, j]
    cv2.imshow('nms', nms_img)
    plt.subplot(235)
    plt.imshow(nms_img)


    #   五：双阈值检测
    # 设置阈值
    lower = sobel_img.mean() * 0.5
    higher = lower * 3
    list = []

    for i in range(1, nms_img.shape[0] - 1):
        for j in range(1, nms_img.shape[1] - 1):
            if nms_img[i, j] > higher:
                nms_img[i, j] = 255             # 大于高阈值保留为255
                list.append([i, j])             # 弱边缘放入列表后续判断
            elif nms_img[i, j] < lower:
                nms_img[i, j] = 0               # 小于低阈值舍弃

    # print(list)
    for i in range(0, len(list) - 1):
        temp_1, temp_2 = list[i][0], list[i][1]
        temp_3 = nms_img[temp_1-1:temp_1+2, temp_2-1:temp_2+2]  # 在选取点的8个邻域范围内 [0,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[2,2]
        if (temp_3[0, 0] > lower) and (temp_3[0, 0] < higher):
            nms_img[temp_1-1, temp_2-1] = 255
        if (temp_3[0, 1] > lower) and (temp_3[0, 1] < higher):
            nms_img[temp_1-1, temp_2] = 255
        if (temp_3[0, 2] > lower) and (temp_3[0, 2] < higher):
            nms_img[temp_1-1, temp_2+1] = 255
        if (temp_3[1, 0] > lower) and (temp_3[1, 0] < higher):
            nms_img[temp_1, temp_2-1] = 255
        if (temp_3[1, 2] > lower) and (temp_3[1, 2] < higher):
            nms_img[temp_1, temp_2+1] = 255
        if (temp_3[2, 0] > lower) and (temp_3[2, 0] < higher):
            nms_img[temp_1+1, temp_2-1] = 255
        if (temp_3[2, 1] > lower) and (temp_3[2, 1] < higher):
            nms_img[temp_1+1, temp_2] = 255
        if (temp_3[2, 2] > lower) and (temp_3[2, 2] < higher):
            nms_img[temp_1+1, temp_2+1] = 255

    # 图片二值化
    for i in range(nms_img.shape[0]):
        for j in range(nms_img.shape[1]):
            if (nms_img[i, j] > 0) and (nms_img[i, j] < 255):
                nms_img[i, j] = 0
    cv2.imshow('canny', nms_img)
    plt.subplot(236)
    plt.imshow(nms_img)

cv2.waitKey(0)
plt.axis('off')
plt.show()

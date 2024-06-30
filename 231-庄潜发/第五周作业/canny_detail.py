"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/23 23:48
"""
import cv2
import numpy as np


def get_gauss_img():
    """
    进行高斯降噪
    通过高斯滤波过滤掉原图的一些噪点,让图片更平滑,让图片的平滑的同时也会让图片更模糊
    ① 通过公式生成高斯核
    ② 进行卷积得到图片
    :return:
    """
    sigma = 2
    dim = 5  # 核为5*5
    # 生成一个序列,后序要将原点(0,0)定位在核中心,//向下取整核的半径, [-2, -1, 0, 1, 2]
    temp = [i - dim // 2 for i in range(dim)]
    # 生成高斯核
    gauss_filter = np.zeros([5, 5])
    n1 = 1 / (2 * np.pi * sigma ** 2)
    n2 = -1 / 2 * sigma ** 2
    for j in range(gauss_filter.shape[0]):
        for i in range(gauss_filter.shape[1]):
            gauss_filter[i, j] = n1 * np.exp(n2 * (temp[i] ** 2 + temp[j] ** 2))  # 0,0 -> -2,-2
    # 高斯核归一化, 将所有核之和设置为1
    gauss_filter = gauss_filter / gauss_filter.sum()
    # 进行高斯滤波
    # 填充图片, 高斯核为5, 则填充5//2 = 2, 保证卷积后的图片大小一致
    pad_img = np.pad(src_img, ((2, 2), (2, 2)), mode='constant')
    # 新建空白图片存放卷积后的图片
    gauss_img = np.zeros(src_img.shape)
    # 进行卷积加权求和
    for j in range(gauss_img.shape[0]):
        for i in range(gauss_img.shape[1]):
            gauss_img[i, j] = np.sum(pad_img[i:i + dim, j:j + dim] * gauss_filter)
    # 如果图片显示白色, 说明格式不对, 将图片格式转换为uint8
    gauss_img = gauss_img.astype(np.uint8)
    cv2.imshow('gauss_img', gauss_img)
    return gauss_img


def get_tudu_img(img):
    """
    求梯度, tanθ
    通过sobel算子得到水平和垂直边缘
    :return:
    """
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    tidu_img = np.zeros(img.shape)
    # 进行sobel卷积
    sobel_x = np.zeros(img.shape)
    sobel_y = np.zeros(img.shape)
    # 填充, 使得卷积后的图片大小一致
    pad_img = np.pad(img, ((1, 1), (1, 1)), mode="constant")
    for j in range(tidu_img.shape[0]):
        for i in range(tidu_img.shape[1]):
            sobel_x[i, j] = np.sum(pad_img[i:i + 3, j:j + 3] * sobel_kernel_x)
            sobel_y[i, j] = np.sum(pad_img[i:i + 3, j:j + 3] * sobel_kernel_y)
            # 计算得到梯度幅值
            tidu_img[i, j] = np.sqrt(sobel_x[i, j] ** 2 + sobel_y[i, j] ** 2)
    # 防止分母为0报错
    sobel_x[sobel_x == 0] = 0.00000001
    tan_xita = sobel_y / sobel_x
    # cv2.imshow('sobel_x', sobel_x.astype(np.uint8))
    # cv2.imshow('sobel_y', sobel_y.astype(np.uint8))
    cv2.imshow('tidu_img', tidu_img.astype(np.uint8))
    return tidu_img, tan_xita


def yizhi_img(tidu_img, tan_xita):
    """
    比较像素点的八邻域,判断当前像素点是否为极大值,不是极大值进行抑制
    :param tidu_img: 梯度图像
    :param tan_xita: tanθ,用来判断象限,一三象限为正,大于
    :return: 进行极大值抑制后的图像
    """
    # 建立空白图像保存抑制后的图像
    yizhi_img = np.zeros(tidu_img.shape)
    # 没有进行填充,所以不取边缘的像素点
    for j in range(1, yizhi_img.shape[0]-1):
        for i in range(1, yizhi_img.shape[1]-1):
            # 取得当前像素点的八邻域
            temp = tidu_img[i-1:i+2, j-1:j+2]
            # 标记为False, 表示都不抑制, 后面进行极大值判断赋值, 最后通过标记确定是否对当前像素点进行抑制
            flag = False
            # 第一象限, 正弦一三象限正, 二四象限4
            if tan_xita[i, j] > 0:  # tanθ>0, >0°
                num1 = (temp[0, 2]-temp[1, 2])/tan_xita[i, j] + temp[1, 2]
                num2 = (temp[2, 0]-temp[1, 0])/tan_xita[i, j] + temp[1, 0]
                if tidu_img[i, j] > num1 and tidu_img[i, j] > num2:
                    flag = True
            elif tan_xita[i, j] >= 1:  # tanθ>1, >45°
                num1 = (temp[0, 2]-temp[0, 1])/tan_xita[i, j] + temp[0, 1]
                num2 = (temp[2, 0]-temp[2, 1])/tan_xita[i, j] + temp[2, 1]
                if tidu_img[i, j] > num1 and tidu_img[i, j] > num2:
                    flag = True
            # 第二象限
            elif tan_xita[i, j] <= -1:  # tanθ<-=1, <=135°
                num1 = (temp[0, 1]-temp[0, 0])/tan_xita[i, j] + temp[0, 1]
                num2 = (temp[2, 1]-temp[2, 2])/tan_xita[i, j] + temp[2, 1]
                if tidu_img[i, j] > num1 and tidu_img[i, j] > num2:
                    flag = True
            elif tan_xita[i, j] < 0:  # tanθ<0, <180°
                num1 = (temp[1, 0]-temp[0, 0])/tan_xita[i, j] + temp[1, 0]
                num2 = (temp[1, 2]-temp[2, 2])/tan_xita[i, j] + temp[1, 2]
                if tidu_img[i, j] > num1 and tidu_img[i, j] > num2:
                    flag = True
            if flag:
                yizhi_img[i, j] = tidu_img[i, j]
    cv2.imshow("yizhi_img", yizhi_img.astype(np.uint8))
    return yizhi_img


def double_threshold(yizhi_img):
    # 双阈值检测,设定一个高阈值和低阈值,大于高阈值则为强边缘取255,小于低阈值则为弱边缘取0
    # 当出现小于强边缘且大于弱边缘的像素值,遍历强边缘的像素值,在八邻域中出现这种情况即可设置为255(连接边缘)
    low_threshold = 70
    high_threshold = 200
    # 生成一个列表保存高阈值像素点,用来判断高阈值像素点八邻域是否有可连接边缘
    temp_list = []
    for j in range(1, yizhi_img.shape[0]):
        for i in range(1, yizhi_img.shape[1]):
            if yizhi_img[i, j] >= high_threshold:
                yizhi_img[i, j] = 255
                # 这里存储的是高阈值像素点坐标
                temp_list.append([i, j])
            elif yizhi_img[i, j] <= low_threshold:
                yizhi_img[i, j] = 0
    # 列表不为空就一直遍历
    while not len(temp_list) == 0:
        # 拿出像素点坐标,获取到八邻域,判断八邻域是否符合条件
        x, y = temp_list.pop()
        temp = yizhi_img[x-1:x+2, y-1:y+2]
        if temp[0, 0] > low_threshold and temp[0, 0] < high_threshold:
            yizhi_img[x-1, y-1] = 255
            temp_list.append([x-1, y-1])
        elif temp[0, 1] > low_threshold and temp[0, 1] < high_threshold:
            yizhi_img[x-1, y] = 255
            temp_list.append([x-1, y])
        elif temp[0, 2] > low_threshold and temp[0, 2] < high_threshold:
            yizhi_img[x-1, y+1] = 255
            temp_list.append([x-2, y])
        elif temp[1, 0] > low_threshold and temp[1, 0] < high_threshold:
            yizhi_img[x, y-1] = 255
            temp_list.append([x, y-1])
        elif temp[1, 2] > low_threshold and temp[1, 2] < high_threshold:
            yizhi_img[x, y+1] = 255
            temp_list.append([x, y+1])
        elif temp[2, 0] > low_threshold and temp[2, 0] < high_threshold:
            yizhi_img[x+1, y-1] = 255
            temp_list.append([x+1, y-1])
        elif temp[2, 1] > low_threshold and temp[2, 1] < high_threshold:
            yizhi_img[x+1, y] = 255
            temp_list.append([x+1, y])
        elif temp[2, 2] > low_threshold and temp[2, 2] < high_threshold:
            yizhi_img[x+1, y+1] = 255
            temp_list.append([x+1, y+1])
    # 二值化
    for j in range(1, yizhi_img.shape[0]):
        for i in range(1, yizhi_img.shape[1]):
            if yizhi_img[i, j] != 0 and yizhi_img[i, j] != 255:
                yizhi_img[i, j] = 0
    cv2.imshow("yizhi2_img", yizhi_img.astype(np.uint8))


if __name__ == '__main__':
    src_img = cv2.imread('../lenna.png', flags=0)
    cv2.imshow("src_img", src_img)
    gauss_img = get_gauss_img()  # 高斯平滑
    tidu_img, tan_xita = get_tudu_img(gauss_img)  # 通过sobel求得梯度幅值, tanθ
    yizhi_img = yizhi_img(tidu_img, tan_xita)  # 通过tanθ来计算出幅值虚拟点,比较大小,小则进行抑制
    double_threshold(yizhi_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

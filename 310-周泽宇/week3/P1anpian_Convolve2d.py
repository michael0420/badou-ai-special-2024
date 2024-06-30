import numpy as np

# 简单的实现二维卷积函数
def Convolve2d_Diy(image, kernel):

    # 获取图像和卷积核的尺寸
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape

    # 计算输出图像的尺寸
    o_h = i_h - k_h + 1 # 这里默认图片是无填充的卷积
    o_w = i_w - k_w + 1 # 所以输出的大小是图像尺寸-卷积核尺寸+1

    output = np.zeros((o_h, o_w))
    temp = np.zeros((k_w, k_h))

    # 对图像的每一个像素进行卷积运算
    for i in range(o_h):
        for j in range(o_w):
            temp = image[i:i+k_h, j:j+k_w] * kernel
            # image[i:i+k_h, j:j+k_w]选取原始图像的子区域 大小和卷积核大小一样
            output[i, j] = np.sum(temp)
            # 将结果矩阵累加存入对应位置

    return output
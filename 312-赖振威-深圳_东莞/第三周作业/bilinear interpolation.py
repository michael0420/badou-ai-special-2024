#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np  # 导入 NumPy 库并使用别名 np
import cv2  # 导入 OpenCV 库

'''
双线性插值的python实现
python implementation of bilinear interpolation

双线性插值是一种在二维空间中用于估算介于已知数据点之间数值的方法。当我们有一个二维网格上的四个已知数据点，并且想要在这些点之间的任意位置估算数值时，可以使用双线性插值。
具体来说，双线性插值通过对四个最近的数据点进行线性插值的方法来估算目标点的数值。这种方法基于以下假设：在一个矩形区域内，两个相对的边界上的数值变化是线性的。
因此，通过在水平方向和垂直方向分别进行一次线性插值，可以得到目标点的估算值。
总的来说，双线性插值是一种简单而有效的方法，适用于许多图像处理和计算机视觉任务中，如图像缩放、纹理映射等。通过在已知数据点之间进行线性插值，可以得到更平滑和连续的结果，提高数据的准确性和可视化效果。
'''

# 这部分是多行注释，用来说明这段代码实现了双线性插值的功能。

def bilinear_interpolation(img, out_dim):
    """
    双线性插值函数
    :param img: 原始图像
    :param out_dim: 输出图像尺寸
    :return: 双线性插值结果图像
    """
    # 获取输入图像尺寸和通道数
    src_h, src_w, channel = img.shape  # 获取原始图像的高度、宽度和通道数
    # print(img.shape) # (512, 512, 3)
    # print("src_h, src_w = ", src_h, src_w)  # 512 512

    # 创建目标图像 create destination image
    dst_h, dst_w = out_dim[1], out_dim[0]  # 获取目标图像的高度和宽度
    # print("dst_h, dst_w = ", dst_h, dst_w)  # 700 700

    # 如果原始图像尺寸与目标图像尺寸相同，则直接返回原图像副本
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    '''
    dtype=np.uint8 表示创建一个数据类型为无符号8位整数的数组。
    在图像处理中，通常使用无符号8位整数来表示图像的像素值，范围从 0 到 255。这种数据类型可以有效地表示图像的亮度值，同时也可以减小内存占用
    '''
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 创建目标图像数组
    # print(dst_img)

    # 计算横向和纵向的缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 计算缩放比例
    # print(scale_x, scale_y)   # 0.7314285714285714 0.7314285714285714

    # 对每个通道进行插值计算
    for i in range(3):  # 遍历三个颜色通道
        for dst_y in range(dst_h):  # 遍历目标图像的高度
            for dst_x in range(dst_w):  # 遍历目标图像的宽度
                # find the origin x and y coordinates of dst image x and y  找到dst图像x和y的原点x和y坐标
                # use geometric center symmetry   使用几何中心对称
                # if use direct way, src_x = dst_x * scale_x   如果使用直接方式，src_x = dst_x * scale_x
                '''
                dst_x 表示目标图像中的 x 坐标，scale_x 是原始图像宽度与目标图像宽度的比例
                首先，我们将目标图像中的 x 坐标加上 0.5，这是为了将目标图像的像素中心对齐到原始图像中（因为插值通常是以像素中心为基准进行的），这里矩阵是0，所以+0.5就可以对齐中心。
                然后乘以缩放比例 scale_x，将目标图像中的坐标映射到原始图像中。
                最后再减去 0.5，是为了将坐标调整回原始图像的像素网格中。

                对于这样的双线性插值计算中的目标图像坐标偏移操作，可能需要更详细地解释一下：
                在双线性插值中，我们通常希望在原始图像的像素之间进行插值计算，以获取目标图像中每个像素的值。

                为了准确地定位目标图像中的像素在原始图像中的位置，我们需要进行一些坐标的调整。
                将目标图像中的坐标加上 0.5 的操作是为了将目标图像的像素中心对齐到原始图像的像素网格中。这种处理方式有助于减小插值误差，使得插值结果更加准确。

                接着乘以缩放比例（即原始图像尺寸与目标图像尺寸的比例），这个操作是为了将目标图像中的坐标映射到原始图像中对应的位置上。
                这样可以确保在原始图像中找到与目标图像像素位置最接近的像素点用于插值计算。

                最后，减去 0.5 的操作是为了将计算得到的坐标再次调整回原始图像的像素网格中，以便准确地确定插值所需的像素点位置。
                综上所述，通过这些操作，我们能够在双线性插值过程中准确地定位目标图像中每个像素在原始图像中的位置，并进行插值计算得到最终的结果。这些步骤有助于提高插值的精度和准确性。
                希望这样的解释能够帮助你理解这部分代码的含义。
                '''
                # 找到目标图像像素在原图像中的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5  # 计算目标图像像素在原始图像中的 x 坐标
                src_y = (dst_y + 0.5) * scale_y - 0.5  # 计算目标图像像素在原始图像中的 y 坐标

                # find the coordinates of the points which will be used to compute the interpolation
                # 找到将用于计算插值的点的坐标

                # 找到原图像用于插值计算的四个点的坐标 来计算插值赋值像素值给目标图像的4个权重参与计算
                # 向下取整， x 方向上的左侧邻近像素的位置
                src_x0 = int(np.floor(src_x))  # 0的 这两行代码是用来找到目标图像中当前像素在原始图像中最近的左上角像素坐标。通过向下取整，找到目标图像像素在原始图像中的左上角位置
                # x 方向上的右侧邻近像素的位置   src_x0 + 1 x 方向上的右侧邻近像素的位置 src_w - 1 是为了确保不会超出原始图像的宽度范围
                src_x1 = min(src_x0 + 1, src_w - 1)  # 1的 这两行代码x,y都+1 是为了找到插值计算中需要使用的另外一个最近的像素坐标，即右下角像素坐标。这里做了边界处理，确保不会超出原始图像的范围
                # src_y0 是根据目标像素的 y 坐标 src_y 向下取整得到的值，表示目标像素在原始图像中 y 方向上的上侧邻近像素的位置。
                src_y0 = int(np.floor(src_y))  # 最近的 y 坐标向下取整
                # src_y1 是根据 src_y0 计算得到的下侧邻近像素的位置，但同时要确保不会超出原始图像的高度范围，即取 src_y0 + 1 和 src_h - 1 中的较小值。
                src_y1 = min(src_y0 + 1, src_h - 1)  # 计算最近的 y+1 坐标

                # calculate the interpolation  计算插值
                '''
                这段代码是进行双线性插值计算，用于在图像中根据已知的坐标位置(src_x, src_y)计算出目标位置(src_x1, src_y1)处的像素值。通过这种插值计算可以获得更加平滑和精确的图像处理效果，常用于图像缩放、旋转等操作中。其中temp0和temp1分别表示目标位置处在不同通道(i)上的插值计算结果。
                '''
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]  # 插值计算
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]  # 插值计算

                # 给创建的dst_img图像指定坐标赋值像素值   整个矩阵三个通道通过BGR挨个赋值
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)  # 完成插值

    return dst_img  # 返回双线性插值后的目标图像

if __name__ == '__main__':
    img = cv2.imread('lenna.png')  # 读取名为 "lenna.png" 的图像
    cv2.imshow('img', img)  # 显示处理后的图像
    cv2.waitKey()  # 等待用户按下任意键关闭窗口

    out_dim = (700, 700)    # 需要输出的图像尺寸

    dst = bilinear_interpolation(img, out_dim)  # 进行双线性插值
    cv2.imshow('bilinear_interpolation', dst)  # 显示处理后的图像
    cv2.waitKey()  # 等待用户按下任意键关闭窗口




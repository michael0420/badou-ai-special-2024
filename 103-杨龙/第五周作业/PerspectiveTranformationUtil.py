# -*- coding:utf-8 -*-

__author__ = 'Young'

import cv2
import numpy as np


def perspective_transformation(img, src, dst, dsize):
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img.copy(), m, dsize)


def find_vertices(img):
    """
    寻找原始坐标点的方式有很多，软件方法，鼠标悬停图像，还有其他，一般是src和dst由需求方给出
    针对软件方法获取原始坐标点，后续可以深入了解下：如何寻找几何图像或不规则图像顶点坐标。
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edged = cv2.Canny(dilate, 30, 120, 3)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    for peak in docCnt:
        peak = peak[0]
        cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    return img


def warp_perspective_matrix(src, dst):
    """
    # 求解透视变换矩阵原理 A*warpmartix=B
    :param src: 原始点坐标集(默认为4个)
    :param dst: 变换后的坐标集(一般和src数据集大小一致)
    :return: 返回透视变换矩阵
    """
    __verify_param(src, dst)
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # 初始化数组A，用于计算矩阵A
    B = np.zeros((2 * nums, 1))  # 初始化数组B
    # 根据公式填充A，B数组
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i, :] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1, :] = B_i[1]
    A = np.mat(A)  # 转换为矩阵
    warp_Matrix = A.I * B  # A的逆矩阵与B相乘，得到warpmartix
    warp_Matrix = np.array(warp_Matrix).T[0]  # 转置矩阵，方便插入Z对应的数据
    warp_Matrix = np.insert(warp_Matrix, warp_Matrix.shape[0], values=1.0, axis=0)
    warp_Matrix = warp_Matrix.reshape((3, 3))
    return warp_Matrix


def __verify_param(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

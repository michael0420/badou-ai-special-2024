# -*- coding: utf-8 -*-
'''@Time: 2024/5/26 10:42

'''
#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:RANSAC_Line.py
#
#Date:20-2-29
#Author:liheng
#Version:V1.0
#============================#

import numpy as np
import random
import math

import cv2

def fitLineRansac(points,iterations=1000,sigma=1.0,k_min=-7,k_max=7):
    """
    RANSAC 拟合2D 直线
    :param points:输入点集,numpy [points_num,1,2],np.float32
    :param iterations:迭代次数
    :param sigma:数据和模型之间可接受的差值,车道线像素宽带一般为10左右
                （Parameter use to compute the fitting score）
    :param k_min:
    :param k_max:k_min/k_max--拟合的直线斜率的取值范围.
                考虑到左右车道线在图像中的斜率位于一定范围内，
                添加此参数，同时可以避免检测垂线和水平线
    :return:拟合的直线参数,It is a vector of 4 floats
                (vx, vy, x0, y0) where (vx, vy) is a normalized
                vector collinear to the line and (x0, y0) is some
                point on the line.
    """
    line = [0,0,0,0]
    points_num = points.shape[0]

    if points_num<2:
        return line

    bestScore = -1
    for k in range(iterations):
        i1,i2 = random.sample(range(points_num), 2)
        p1 = points[i1][0]
        p2 = points[i2][0]

        dp = p1 - p2 #直线的方向向量
        dp *= 1./np.linalg.norm(dp) # 除以模长，进行归一化

        score = 0
        a = dp[1]/dp[0]
        if a <= k_max and a>=k_min:
            for i in range(points_num):
                v = points[i][0] - p1
                dis = v[1]*dp[0] - v[0]*dp[1]#向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
                # score += math.exp(-0.5*dis*dis/(sigma*sigma))误差定义方式的一种
                if math.fabs(dis)<sigma:
                    score += 1
        if score > bestScore:
            line = [dp[0],dp[1],p1[0],p1[1]]
            bestScore = score

    return line



if __name__ == '__main__':
    image = np.ones([720,1280,3],dtype=np.ubyte)*125

    # 以车道线参数为(0.7657, -0.6432, 534, 548)生成一系列点
    k = -0.6432 / 0.7657
    b = 548 - k * 534

    points = []
    for i in range(360,720,10):
        point = (int((i-b)/k),i)
        points.append(point)

    # 加入直线的随机噪声
    for i in range(360,720,10):
        x = int((i-b)/k)
        x = random.sample(range(x-10,x+10),1)
        y = i
        y = random.sample(range(y - 30, y + 30),1)

        point = (x[0],y[0])
        points.append(point)

    # 加入噪声
    for i in range(0,720,20):
        x = random.sample(range(1, 640), 1)
        y = random.sample(range(1, 360), 1)
        point = (x[0], y[0])
        points.append(point)

    for point in points:
        cv2.circle(image,point,5,(0,0,0),-1)


    points = np.array(points).astype(np.float32)
    points = points[:,np.newaxis,:]

    # RANSAC 拟合
    if 1:
        [vx, vy, x, y] = fitLineRansac(points,1000,10)
        k = float(vy) / float(vx)  # 直线斜率
        b = -k * x + y

        p1_y = 720
        p1_x = (p1_y-b) / k
        p2_y = 360
        p2_x = (p2_y-b) / k

        p1 = (int(p1_x),int(p1_y))
        p2 = (int(p2_x), int(p2_y))

        cv2.line(image,p1,p2,(0,255,0),2)

    # 最小二乘法拟合
    if 1:
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.1, 0.01)
        k = float(vy) / float(vx)  # 直线斜率
        b = -k * x + y

        p1_y = 720
        p1_x = (p1_y - b) / k
        p2_y = 360
        p2_x = (p2_y - b) / k

        p1 = (int(p1_x), int(p1_y))
        p2 = (int(p2_x), int(p2_y))

        cv2.line(image, p1, p2, (0, 0, 255), 2)


    cv2.imshow('image',image)
    cv2.waitKey(0)


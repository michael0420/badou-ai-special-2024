# -*- coding: utf-8 -*-

import cv2
import numpy as np

def get_slider_gap(gap_img_path):
    try:
        # 读取图片
        image = cv2.imread(gap_img_path)

        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)   # 50和150是Canny算法的两个阈值参数
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('edges\n',edges)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('contours\n',contours)
        ''''
        代码使用了OpenCV的 findContours() 函数来查找图像中的轮廓。让我来解释一下：
        edges：这是通过Canny边缘检测算法得到的图像边缘信息，它是一个二值化的图像，其中非零像素表示边缘，零像素表示背景。
        cv2.RETR_EXTERNAL：这是轮廓检索模式参数，指定了轮廓的检索模式。RETR_EXTERNAL 表示只检测外部轮廓，即不检测轮廓内部的子轮廓。
        cv2.CHAIN_APPROX_SIMPLE：这是轮廓逼近方法参数，指定了轮廓的逼近方法。CHAIN_APPROX_SIMPLE 表示对轮廓中的冗余点进行压缩，只保留终点坐标，如一个矩形轮廓只需四个顶点坐标来表示。
        findContours() 函数返回两个值，第一个是轮廓信息的列表，每个轮廓都是一个点集，第二个值是层析结构。在这里，我们使用了下划线 _ 来表示我们对第二个返回值不感兴趣，因此我们没有将其赋值给任何变量。
        所以，这一行代码的作用是在边缘图像 edges 中找到轮廓，并将轮廓信息存储在 contours 变量中。
        findContours() 函数返回两个值：一个是轮廓信息的列表，另一个是层析结构。然而，我们只对轮廓信息感兴趣，而对层析结构不感兴趣。因此，我们可以使用下划线 _ 来标记我们不关心的返回值，以示清楚
        请注意，轮廓是通过像素的连续性来定义的，因此轮廓上的点的顺序可能并不是按照从左到右或从上到下的顺序排列的。如果需要按照特定顺序处理轮廓上的点，可能需要进行额外的排序操作。
        '''

        # 找到最长的轮廓（即滑块轮廓）
        max_contour = max(contours, key=cv2.contourArea)
        print('max_contour\n', max_contour)
        '''
        这一行代码使用了 Python 的 max() 函数来找到具有最大面积的轮廓，并将其存储在变量 ax_contour 中。让我来解释一下：
        contours 是之前通过 findContours() 函数找到的轮廓信息的列表。
        cv2.contourArea 是一个函数，用于计算轮廓的面积。
        max() 函数用于找到列表中的最大值。在这里，我们使用了 key 参数来指定比较的依据，即根据轮廓的面积来进行比较。
        所以，这一行代码的作用是从 contours 列表中找到面积最大的轮廓，并将其赋值给变量 ax_contour。
        '''

        # 创建空白图像
        max_contour_image = np.zeros_like(image)
        # 将最长轮廓绘制到空白图像上
        cv2.drawContours(max_contour_image, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        # 显示绘制好轮廓的图像
        cv2.imshow('max_contour', max_contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        轮廓数组并不是一个完整的图像，而是仅包含轮廓上的点坐标，所以在显示时可能无法正常展示出预期的效果。
        在使用 OpenCV 的 cv2.imshow() 函数显示图像时，需要传入一个图像作为参数。而轮廓（max_contour）是由一系列的点坐标构成的数组，并不是一个完整的图像。
        因此，为了能够使用 cv2.imshow() 来显示轮廓，我们需要先创建一个与原始图像大小相同的空白图像，然后使用 cv2.drawContours() 函数将轮廓绘制到空白图像上。
        这样就得到了一个包含了轮廓形状的图像，可以直接传给 cv2.imshow() 函数进行显示。
        通过创建空白图像并绘制轮廓，我们可以更好地可视化轮廓的形状和位置，而不是简单地显示一个轮廓的点坐标。这样可以更方便地观察和理解轮廓的特征。
        '''

        # 获取滑块轮廓的边界框
        x, _, _, _ = cv2.boundingRect(max_contour)
        print('x, _, _,\n',x, _, _,)

        # 计算滑块缺口的左边缘到图片左边缘的距离
        slider_gap = x

        print(f'slider_gap检测为{slider_gap}')

        return slider_gap

    except Exception as e:
        print(f'get_slider_gap方法报错：{e}')

gap_img_path = 'capcha.jpg'
get_slider_gap(gap_img_path)


'''
cv2.approxPolyDP() 多边形逼近
作用:
对目标图像进行近似多边形拟合，使用一个较少顶点的多边形去拟合一个曲线轮廓，要求拟合曲线与实际轮廓曲线的距离小于某一阀值。

函数原形：
cv2.approxPolyDP(curve, epsilon, closed) -> approxCurve

参数：
curve ： 图像轮廓点集，一般由轮廓检测得到
epsilon ： 原始曲线与近似曲线的最大距离，参数越小，两直线越接近
closed ： 得到的近似曲线是否封闭，一般为True

返回值：
approxCurve ：返回的拟合后的多边形顶点集。
'''

import cv2

# 读取图像
img = cv2.imread('photo1.jpg')

# 将图像转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图进行高斯模糊处理，卷积核大小为5x5，标准差为0
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 创建一个矩形结构元素，大小为3x3
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# 对模糊后的图像进行膨胀操作
dilate = cv2.dilate(blurred, kernel)

# 使用Canny边缘检测算法检测图像中的边缘，阈值范围为30到120，孔径大小为3
edged = cv2.Canny(dilate, 30, 120, 3)

# 检测图像中的轮廓，使用外部轮廓检索模式和简单轮廓近似方法
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 根据OpenCV版本获取轮廓
cnts = cnts[0]

# 初始化目标轮廓为None
docCnt = None

# 如果检测到至少一个轮廓
if len(cnts) > 0:
    # 根据轮廓面积从大到小排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 遍历排序后的轮廓
    for c in cnts:
        # 计算轮廓的周长
        peri = cv2.arcLength(c, True)
        # 使用多边形逼近方法拟合轮廓，参数为轮廓的2%精度
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # 如果拟合后的多边形有4个顶点
        if len(approx) == 4:
            # 认为找到了目标轮廓（例如一张纸）
            docCnt = approx
            break

# 在原图像上标记找到的四个顶点
if docCnt is not None:
    for peak in docCnt:
        # 提取顶点坐标
        peak = peak[0]
        # 在顶点处画一个红色圆圈
        cv2.circle(img, tuple(peak), 10, (255, 0, 0))

# 显示标记后的图像
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

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

img = cv2.imread('photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
edged = cv2.Canny(dilate, 30, 120, 3)            # 边缘检测

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
cnts = cnts[0]
#if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 根据轮廓面积从大到小排序
    for c in cnts:
        peri = cv2.arcLength(c, True)                           # 计算轮廓周长
        approx = cv2.approxPolyDP(c, 0.02*peri, True)           # 轮廓多边形拟合
        # 轮廓为4个点表示找到纸张
        if len(approx) == 4:
            docCnt = approx
            break

for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))

cv2.imshow('img', img)
cv2.waitKey(0)

'''
img = cv2.imread('photo.jpg'): 读取名为 "photo.jpg" 的图像文件。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY): 将彩色图像转换为灰度图像。
blurred = cv2.GaussianBlur(gray, (5, 5), 0): 对灰度图像进行高斯模糊处理，这一步通常用于去除图像中的噪点。
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))): 对模糊后的图像进行膨胀操作，这一步可以使边缘变得更加连续。
edged = cv2.Canny(dilate, 30, 120, 3): 使用 Canny 边缘检测算法检测图像中的边缘。
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE): 寻找图像中的轮廓。
cnts = cnts[0]: 由于返回值的格式可能会有所不同，这里根据 OpenCV 的版本选择相应的格式。
docCnt = None: 初始化变量，用于存储找到的纸张轮廓。
if len(cnts) > 0:: 如果找到了轮廓，则执行下面的操作。
cnts = sorted(cnts, key=cv2.contourArea, reverse=True): 根据轮廓的面积从大到小对轮廓进行排序。
for c in cnts:: 遍历排序后的轮廓。
peri = cv2.arcLength(c, True): 计算轮廓的周长。
approx = cv2.approxPolyDP(c, 0.02*peri, True): 对轮廓进行多边形逼近，这一步用于简化轮廓的形状。
if len(approx) == 4:: 如果逼近后的轮廓有四个点，说明可能找到了纸张的边缘。
docCnt = approx: 将找到的四个角点保存到变量 docCnt 中。
for peak in docCnt:: 遍历四个角点。
peak = peak[0]: 由于 approxPolyDP 返回的是一个三维数组，我们只需要第一个维度的值，即每个点的坐标。
cv2.circle(img, tuple(peak), 10, (255, 0, 0)): 在图像上绘制蓝色的圆圈，表示检测到的纸张角点。
cv2.imshow('img', img): 显示带有纸张角点的图像。
cv2.waitKey(0): 等待用户按下键盘上的任意键，然后关闭图像窗口。
这段代码的主要功能是使用 Canny 边缘检测算法检测图像中的边缘，然后根据轮廓找到可能代表纸张的区域，并在纸张的四个角点位置绘制蓝色圆圈。
'''
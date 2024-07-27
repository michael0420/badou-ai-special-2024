import cv2  # 导入OpenCV库
# import imutils  # 如果需要，可以导入imutils库（这里注释掉）

'''
### 代码逻辑和作用
1. **读取并预处理图像**：
   - 读取图像并转换为灰度图像。
   - 对灰度图像进行高斯模糊以减少噪声。
   - 对模糊后的图像进行膨胀操作以增强边缘。
   - 使用Canny算法进行边缘检测。

2. **轮廓检测和多边形逼近**：
   - 检测图像中的轮廓。
   - 对检测到的轮廓按面积进行排序，从大到小。
   - 遍历排序后的轮廓，使用多边形逼近算法拟合轮廓。
   - 如果拟合后的多边形有4个顶点，则认为找到了目标（纸张）。

3. **绘制顶点**：
   - 遍历找到的目标轮廓的顶点，在图像上绘制蓝色圆圈标记顶点位置。

4. **显示结果**：
   - 显示标记了顶点的图像，并等待按键按下以关闭窗口。

这个代码主要用于检测图像中的矩形物体（如纸张），并标记其顶点。

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

img = cv2.imread('photo1.jpg')  # 读取图像，文件名为'photo1.jpg'
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将读取的彩色图像转换为灰度图像
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 对灰度图像进行高斯模糊处理，内核大小为5x5，标准差为0
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 对模糊后的图像进行膨胀操作，使用3x3的矩形结构元素
edged = cv2.Canny(dilate, 30, 120, 3)  # 边缘检测，使用Canny算法，阈值为30和120，使用3x3的Sobel算子

# 轮廓检测，使用cv2.RETR_EXTERNAL模式（只检测外轮廓），cv2.CHAIN_APPROX_SIMPLE方法（压缩水平、垂直和对角线段，保留终点）
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]  # OpenCV 2和3的兼容性处理，取第一个元素
# if imutils.is_cv2() else cnts[1]  # 判断是OpenCV2还是OpenCV3，OpenCV4可以忽略

docCnt = None  # 初始化docCnt变量，用于存储找到的纸张轮廓

if len(cnts) > 0:  # 如果检测到至少一个轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 根据轮廓面积从大到小排序
    for c in cnts:  # 遍历排序后的轮廓
        peri = cv2.arcLength(c, True)  # 计算轮廓的周长，True表示轮廓是闭合的
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 多边形逼近，epsilon为周长的0.02倍，True表示闭合的多边形
        # 如果逼近后的多边形有4个顶点，说明找到一个矩形（纸张）
        if len(approx) == 4:
            docCnt = approx  # 将找到的纸张轮廓存储到docCnt
            break  # 找到后退出循环

# 遍历纸张轮廓的每一个顶点
for peak in docCnt:
    peak = peak[0]  # 获取顶点的坐标
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))  # 在图像上画圆，圆心为顶点坐标，半径为10，颜色为蓝色（BGR格式）

cv2.imshow('img', img)  # 显示图像
cv2.waitKey(0)  # 等待按键按下，确保图像窗口不会立即关闭



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


img = cv2.imread(r"E:\image\photol.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 用于执行膨胀操作,扩张图像中的前景对象（如白色区域或亮区）
dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# Canny边缘检测
edged = cv2.Canny(dilate, 30, 120, 3)

# 寻找轮廓
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cnts = cnts[0] 提取出来的值是一个 Python 列表，列表中的每个元素都是一个 numpy 数组，代表了一组轮廓上的点坐标序列。
# 在 OpenCV 中，轮廓是以点的序列形式存储的，
cnts = cnts[0]

#if imutils.is_cv2() else cnts[1]  # 判断是opencv2还是opencv3

docCnt = None

if  len(cnts) > 0:          # 检查是否找到了轮廓（cnts 是轮廓的列表）
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # 轮廓按面积排序
    
    for i in cnts:
    
        # 使用 cv2.arcLength() 函数计算轮廓 c 的周长。第二个参数 True 表示轮廓是封闭的
        peri = cv2.arcLength(c, true)
        
        # 对轮廓进行多边形拟合简化，目的是减少轮廓上的点数，同时保持轮廓的基本形状。参数 0.02*peri 是一个阈值，用于决定近似精度，这里设置为轮廓周长的2%。第三个参数再次表明轮廓是封闭的。
        approx = cv2.approxPolyDP(c, 0,02*peri, True)
        
        # 检查简化后的轮廓 approx 是否有4个顶点
        if len(approx) == 4:
            docCnt = approx
            break
            
for peak in docCnt:
    peak = peak[0]
    cv2.circle(img, tuple(peak), 10, (255, 0, 0))
    
cv2.imshow('img', img)
cv2.waitKey(0)   

import cv2
#import imutils

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

img = cv2.imread('photo1.jpg')
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
cv2.GaussianBlur 是 OpenCV 中用于对图像进行高斯模糊处理的函数。高斯模糊是一种常见的图像处理操作，用于降低图像的噪声和细节，使图像变得更加平滑。
这个函数的基本语法如下：
    dst = cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
    src：输入的图像，可以是灰度图像或彩色图像。
    ksize：高斯核的大小，表示高斯核的宽和高，必须是正奇数，例如 (5, 5)、(9, 9) 等。
    sigmaX：X 方向上的高斯核标准差。
    sigmaY：Y 方向上的高斯核标准差，如果设为 0，则会自动设为 sigmaX 的值，如果 sigmaX 和 sigmaY 都是 0，它们会根据核函数的大小自动计算。
    borderType：可选参数，表示边界扩充的方式，常用的有 cv2.BORDER_CONSTANT、cv2.BORDER_REPLICATE 等。
    这个函数会对输入的图像进行高斯模糊处理，并返回处理后的图像。高斯模糊的效果取决于 ksize 和 sigmaX、sigmaY 的取值，可以通过调整这些参数来控制模糊的程度。
    
cv2.dilate 是 OpenCV 中用于图像膨胀（Dilation）操作的函数。膨胀是图像处理中常用的形态学操作之一，用于增加图像中物体的像素点。
这个函数的基本语法如下：
    dst = cv2.dilate(src, kernel, anchor, iterations, borderType, borderValue)
    src：输入的图像，可以是灰度图像或彩色图像。
    kernel：膨胀操作的核，用于确定膨胀的形状和大小。
    anchor：核的锚点，默认为 (-1, -1)，表示核的中心点。
    iterations：膨胀操作的迭代次数，表示对图像进行膨胀操作的次数。
    borderType：可选参数，表示边界扩充的方式，常用的有 cv2.BORDER_CONSTANT、cv2.BORDER_REPLICATE 等。
    borderValue：可选参数，表示边界填充的值。
    这个函数会对输入的图像进行膨胀操作，并返回处理后的图像。膨胀操作会使图像中的物体区域扩张，增加其像素点的数量。膨胀操作通常用于图像处理中的形态学操作，例如去除噪声、连接物体等。

cv2.Canny 是 OpenCV 中用于进行边缘检测的函数，它能够帮助找到图像中的边缘部分。
这个函数的基本语法如下：
    edges = cv2.Canny(image, threshold1, threshold2, apertureSize, L2gradient)
    image：输入的灰度图像。
    threshold1：第一个阈值，用于边缘检测中的强边缘。
    threshold2：第二个阈值，用于边缘检测中的弱边缘。
    apertureSize：Sobel 算子的孔径大小，常用的值为 3、5、7。
    L2gradient：一个布尔值，表示计算图像梯度幅值的方式，如果设为 True，则使用更精确的 L2 范数进行计算。
    这个函数会对输入的灰度图像进行边缘检测，并返回一个包含边缘信息的二值图像。边缘检测是图像处理中的重要操作，能够帮助识别图像中的边缘结构，常用于目标检测、图像分割等领域。
cv2.findContours 是 OpenCV 中用于在二值图像中查找轮廓的函数。它能够帮助找到图像中的连通对象的边界信息。
    
cv2.findContours 是 OpenCV 中用于在二值图像中查找轮廓的函数。它能够帮助找到图像中的连通对象的边界信息。
    这个函数的基本语法如下：
    contours, hierarchy = cv2.findContours(image, mode, method)
    image：输入的二值图像，通常是经过阈值处理或者边缘检测后得到的图像。
    mode：轮廓检索模式，指定轮廓的检索模式，如 cv2.RETR_EXTERNAL 表示只检测外部轮廓，cv2.RETR_LIST 表示检测所有轮廓等。
    method：轮廓逼近方法，指定轮廓的逼近方法，如 cv2.CHAIN_APPROX_SIMPLE 表示只保留终点坐标，cv2.CHAIN_APPROX_NONE 表示保留所有的轮廓点等。
    这个函数会在输入的二值图像中查找轮廓，并返回一个包含轮廓信息的列表以及一个层级信息。轮廓信息通常以一组点的形式给出，可以用于后续的形状分析、对象检测等操作。
        
'''
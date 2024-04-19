import cv2
import numpy as np
 
img = cv2.imread("lenna.png", 0)
if img is None:
    print("Failed to load the image.")
    exit()

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# 对图像img应用水平方向的Sobel算子进行边缘检测。
# cv2.CV_16S表示将输出图像的数据类型设置为16位有符号整数。
# 1和0分别表示在x方向上应用一阶导数,在y方向上不应用导数。

y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# 对图像img应用垂直方向的Sobel算子进行边缘检测。
# 0和1分别表示在x方向上不应用导数,在y方向上应用一阶导数。

'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

absX = cv2.convertScaleAbs(x)
# 将x中的值转换为绝对值,并将数据类型转换为8位无符号整数。
# 这是因为Sobel算子的输出可能包含负值,而图像像素值通常是非负的。
# 转换后的结果存储在变量absX中。
absY = cv2.convertScaleAbs(y)
# 将y中的值转换为绝对值,并将数据类型转换为8位无符号整数。
# 转换后的结果存储在变量absY中。

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# 将absX和absY两个图像以加权的方式组合起来。
# 第二个参数0.5表示absX的权重,第四个参数0.5表示absY的权重。
# 第五个参数0表示额外的偏置值,这里设为0。
# 组合后的结果存储在变量dst中。
 
cv2.imshow("absX", absX)
cv2.imshow("absY", absY)
 
cv2.imshow("Result", dst)
 
cv2.waitKey(0)
cv2.destroyAllWindows()

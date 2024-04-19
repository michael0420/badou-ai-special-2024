import cv2
import numpy as np

img = cv2.imread('lenna.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #转为灰度图
'''
Sobel函数求完导数后会有负值，还有会大于255的值。正常的 np.uint8会截断在255以外的值
因此使用cv2.CV2_16S,保存16位有符号的数据类型
'''
sobel_x = cv2.Sobel(img,cv2.CV_16S,1,0) #1,0代表x轴
sobel_y = cv2.Sobel(img,cv2.CV_16S,0,1) #0,1代表y轴
'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

dst_x = cv2.convertScaleAbs(sobel_x)
dst_y = cv2.convertScaleAbs(sobel_y)

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''
dst = cv2.addWeighted(dst_x,0.5,dst_y,0.5,0)

cv2.imshow('dst_x:',dst_x)
cv2.imshow('dst_y:',dst_y)
cv2.imshow('dst:',dst)

cv2.waitKey(0)
cv2.destroyAllWindows()
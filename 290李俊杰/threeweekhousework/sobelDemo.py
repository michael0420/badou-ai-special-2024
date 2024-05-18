# 【第三周作业】
#
#  3.实现sobel边缘检测

import cv2
import matplotlib.pyplot as plt
'''
cv2.imread()有两个参数，第一个参数filename是图片路径，
第二个参数flag表示图片读取模式，共有三种：
cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
cv2.IMREAD_UNCHANGED：包括alpha(包括透明度通道)，可以直接写-1
'''
# 以什么方式读图就返回什么类型图片
# 0代表返回的灰度图
img=cv2.imread("lenna.png",0)

# cv2.imshow("img",img)#显示的灰度图
# cv2.waitKey(0)
'''
Sobel函数求完导数后会有负值，还有会大于255的值。
而原图像是uint8，即8位无符号数(范围在[0,255])，所以Sobel建立的图像位数不够，会有截断。
因此要使用16位有符号的数据类型，即cv2.CV_16S。
'''
# 卷积函数Sobel,16位有符号数据转换参数cv2.CV_16S
# 1,0代表做横向卷积。0,1代表做纵向卷积
x=cv2.Sobel(img,cv2.CV_16S,1,0)
y=cv2.Sobel(img,cv2.CV_16S,0,1)

'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''
dstx=cv2.convertScaleAbs(x)
dsty=cv2.convertScaleAbs(y)

'''
由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
。其函数原型为：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''

dstimg=cv2.addWeighted(dstx,0.5,dsty,0.5,0)
cv2.imshow("dstx",dstx)
cv2.imshow("dsty",dsty)
cv2.imshow("dstimg",dstimg)
cv2.waitKey(0)

# # 用plt方式实现
# dstimg=cv2.cvtColor(dstimg,cv2.COLOR_BGR2RGB)
# plt.subplot(221)
# plt.imshow(dstimg)
# plt.show()
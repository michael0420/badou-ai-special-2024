import cv2
import numpy as np

#第一步：读取图片;
img = cv2.imread("lenna.png",0)  # 0:表示以灰度图像方式加载图像，只包含灰度信息

'''
Sobel算子通常用于边缘检测，尤其是在图像的梯度计算中。
虽然它通常用于灰度图像，但它也可以应用于彩色图像的每个通道。具体来说，可以对RGB图像的每个通道分别应用Sobel算子，然后将结果合并以得到最终的边缘检测结果。

但需要注意的是，直接在彩色图像上应用Sobel算子可能会导致边缘检测结果受到颜色信息的影响，
因此在某些情况下，可能会首先将彩色图像转换为灰度图像，然后应用Sobel算子进行边缘检测，以获得更好的结果。
'''



'''
Sobel函数求完导数之后会有负值，还会有大于255的值。
而原图像是uint8,即8位无符号数，范围在【0，255】，所以Sobel函数所建立的图像位数不够，会有截断。
因此要使用16位有符号数的数据类型，即:cv2.CV_16S。
'''
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
'''
这行代码使用了 OpenCV 中的 `cv2.Sobel()` 函数来应用 Sobel 算子进行图像边缘检测。
具体参数解释如下：

- `img`：要进行边缘检测的输入图像。
- `cv2.CV_16S`：指定输出图像的数据类型。在这里，`cv2.CV_16S` 表示输出的图像将以 16 位有符号整数的形式存储。这是因为 Sobel 算子的计算可能会导致像素值溢出，因此需要使用更大的数据类型来保存结果，以避免信息损失。
- `1, 0`：指定 Sobel 算子的水平和垂直方向的导数阶数。在这里，`1` 表示水平方向的导数，`0` 表示垂直方向的导数。这意味着应用水平方向的 Sobel 算子来检测图像中的垂直边缘。

根据以上参数，该行代码的作用是对输入图像 `img` 应用 Sobel 算子进行水平方向的边缘检测，并将结果保存在一个 16 位有符号整数的图像中。
'''




'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)






'''
然后：

由于Sobel函数是在两个方向计算的，最后还需要用：cv2.addWeighted()函数将x,y方向上的计算的值组合起来。
其函数原型是：
dest = cv2.addweighted(src1,slpha,src2,beta,gamma[,[dst[,dtype]])
其中alpha:是第一个元素的权重，bata是第二个元素的权重。gamma是加到最后结果上的一个值。
'''
dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

cv2.imshow("absX",absX)
cv2.imshow("absY",absY)

cv2.imshow("result",dst)
cv2.waitKey(0)

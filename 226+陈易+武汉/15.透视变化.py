import cv2
import numpy as np

img = cv2.imread("photo1.jpg")  # 读图
result3 = img.copy()            # 复制原图

"""
注意这里src和dst的输入不是图像，而是图像对应的顶点坐标
"""
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

# 生成透视变换矩阵warpMatrix，进行透视变换,参数1：原图上的点  参数2：目标图上的点
m = cv2.getPerspectiveTransform(src,dst)   # 获取逆透视变换矩阵函数warpMatrix的各参数(a11~a33)
print("warpMatrix:\n",m)

"""
warpPerspective函数，使指定的矩阵变化源图像.
cv.warpPerspective (InputArray src, OutputArray dst, dsize, int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar &borderValue=Scalar())
参数说明如下：
InputArray src:输入图像
OutputArray dst：输出大小为dsize
dsize：目标图像的大小，以元组形式表示，例如(width, height)
  ***以下为默认值***
flags： 插值方法的标志，用于指定插值方法，默认为cv2.INTER_LINEAR
borderMode： 边界模式，用于指定超出边界的像素处理方式，默认为cv2.BORDER_CONSTANT
borderValue： 当边界模式为cv2.BORDER_CONSTANT时，用于指定边界像素的值，默认为0。
"""
result = cv2.warpPerspective(result3,m,(337,488))
cv2.imshow("src",img)
cv2.imshow("result",result)
cv2.waitKey(0)
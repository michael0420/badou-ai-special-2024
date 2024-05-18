""""
@author:lpj
彩色图像的 灰度化、二值化
"""

# 常用视觉库
"""
 skimage即是Scikit-Image。基于python脚本语言开发的数字图片处理包，比如PIL,Pillow, opencv, scikit-image等。
 PIL和Pillow只提供最基础的数字图像处理，功能有限；opencv实际上是一个c++库，只是提供了python接口，更新速度非常慢。
 scikit-image是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，正好与matlab一样，因此，我们最终选择scikit-image进行数字图像处理。
"""
from skimage.color import rgb2gray

""""
 NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ---灰度化---
img = cv2.imread("D:\cv_workspace\picture\lenna.png")  # 绝对路径 opencv读进来的图片是BGR 需要转成 RGB
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 生成和当前图片大小([h,w])、类型(dtype)一样的单通道(zeros,全0矩阵)图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出high和wide中的BGR坐标     m为彩色图片中 当前像素点的值 [ 125 137 226]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化成gray坐标并赋值给新图像  img_gray[i, j]为灰度图中当前像素点的值 126
print(m)
print(img_gray[i, j])
print("image show gray[i,j]: %s" % img_gray)
cv2.imshow("image show gray", img_gray)
""""
waitkey控制着imshow的持续时间，当imshow之后不跟waitkey时，相当于没有给imshow提供时间展示图像，所以只有一个空窗口一闪而过。
添加了waitkey后，哪怕仅仅是cv2.waitkey(1),也能截取到一帧的图像。所以cv2.imshow后边是必须要跟cv2.waitkey的。
"""
cv2.waitKey(10000)  # 显示10s




plt.subplot(221)  # 使用plt.subplot来创建小图,plt.subplot(221)表示将整个图像窗口分为2行2列,当前位置为1
img = plt.imread("D:\cv_workspace\picture\lenna.png")
# img = cv2.imread("lenna.png", False)
"""
plt.imshow()是Matplotlib中的一个函数，用于显示图像。它可以传递一个二维或三维数组作为image参数， 并将图像数据显示为图形，并对图像进行不同的可视化设置。
cmap：颜色设置。常用的值有’viridis’、‘gray’、'hot’等。可以通过plt.colormaps()查看可用的颜色映射。
aspect：调整坐标轴。这将根据图像数据自动调整坐标轴的比例。常用的值有’auto’、'equal’等。设置为’auto’时会根据图像数据自动调整纵横比，而设置为’equal’时则会强制保持纵横比相等。
interpolation：插值方法。它定义了图像在放大或缩小时的插值方式。常用的值有’nearest’、‘bilinear’、'bicubic’等。较高的插值方法可以使图像看起来更平滑，但计算成本更高。
alpha：透明度。它允许您设置图像的透明度，取值范围为0（完全透明）到1（完全不透明）之间。
vmin和vmax：用于设置显示的数据值范围。当指定了这两个参数时，imshow()将会根据给定的范围显示图像，超出范围的值会被截断显示。
"""
plt.imshow(img)
print("---image lenna----")
print(img)
#  plt.show()  # 画布展示


#  灰度化  直接调用Api 将图转化成灰度图
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
# plt.show()



# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()





'''
image.shape[:3] 表示取彩色图片的高、宽、通道 
具体的解释为：
  1)输出 height，即获取图像的高度，图像的高度为shape的第一个值（维度）------>print(image.shape[0])
  2)输出 width,获取图像的宽度，图像的宽度为shape的第二个值（维度）print(image.shape[1])
  3)输出通道数，彩色图像为3，灰度图像报错，shape没有第三个参数，只有(height,width)
              获取图像通道数目，图像的通道数目为shape的第三个值（维度）,加载进来的图像都有三个通道，三个通道是图像的BGR print(image.shape[2])
  备注：
      OpenCV中图像读入的数据格式是numpy的ndarray数据格式。是BGR格式，取值范围是[0,255].
'''

'''
np.zeros() 函数是 NumPy 数组库的一部分，用于生成元素全部为 0 的数组
numpy.zeros(shape, dtype=float, order='C')
     shape: 这是一个必需参数，指定了数组的维度。例如，shape=3 创建一个长度为 3 的一维数组，而 shape=(2,3) 则创建一个 2x3 的二维数组。
     dtype: 这个参数允许用户指定数组中元素的数据类型。常见类型包括 numpy.int32, numpy.float64 等。如果不指定，NumPy 默认使用 float64 类型。
     order: 虽然不常用，这个参数允许高级用户优化他们的数组布局，以便更高效地进行特定的数学运算。大多数情况下，默认的 'C' 顺序就足够了。
'''

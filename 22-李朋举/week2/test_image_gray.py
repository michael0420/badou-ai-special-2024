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
'''
    `cv2.imread` 是 OpenCV 库中的一个函数，用于从文件加载图像。
       入参：cv2.imread(path, flags=cv2.IMREAD_COLOR)
            其中，`path` 是图像文件的路径，`flags` 是一个可选参数，用于指定图像的读取方式。常用的取值如下：
                - `cv2.IMREAD_COLOR`：加载彩色图像（默认）
                   1：彩色模式（Color），图像以彩色形式加载，每个像素由红、绿、蓝三个分量组成，每个分量的值在 0 到 255 之间。
                - `cv2.IMREAD_GRAYSCALE`：加载灰度图像  
                   0：灰度模式（Grayscale），图像以灰度形式加载，每个像素的值在 0 到 255 之间。
                - `cv2.IMREAD_UNCHANGED`：加载原始图像，不进行任何转换
                   -1：原始模式（Original），图像以原始格式加载，不进行任何转换。
       反参：cv2.imread 函数返回的是一个 `numpy.ndarray` 对象以多维数组的形式保存图片信息，其中包含了图像的像素数据。
            【img.shape】可以通过函数返回的`numpy.ndarray`对象的`shape`属性来获取图像的格式和大小，
                           返回的值为 `(height, width, channels)`，其中前两维表示图片的像素坐标 `height` 和 `width` 分别表示图像的高度和宽度，
                           最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定`channels`表示图像的通道数（通常为 1 或 3，分别表示灰度图像和彩色图像）。
            【img.dtype】可以通过函数返回的`numpy.ndarray`对象的`dtype`属性来获取图像数据的类型和精度，关于图像数据的类型和精度，
                         `cv2.imread`函数通常会返回以下数据类型之一：
                            - 8 位整数（`np.uint8`）：表示每个像素的值范围为 0 到 255，这是最常见的图像数据类型之一，适用于大多数图像格式。
                            - 16 位整数（`np.uint16`）：表示每个像素的值范围为 0 到 65535，这种类型通常用于一些特殊的图像格式或需要更高精度的应用。
                            - 32 位整数（`np.uint32`）：表示每个像素的值范围为 0 到 4294967295，这种类型也比较常见，尤其是在一些高清图像或需要更高精度的应用中。
                            - 浮点数（`np.float32`或`np.float64`）：表示每个像素的值为浮点数，这种类型通常用于一些科学计算或需要更高精度的应用。
                    np.uint8(RGB24格式):
                        RGB三个色彩通道，每个通道有8位的数据，等级(灰阶)是0～255共256级(2^8)，即色精度为8位，最后通过rgb三个通道加色原理表示，所以一共是24位.
                        BGR色彩空间中第1个8位（第1个字节）存储的是蓝色组成信息（Blue component），第2个8位（第 2 个字节）存储的是绿色组成信息（Green component），
                        第3个8位（第3个字节）存储的是红色组成信息（Red component）。同样，其第 4 个、第 5 个、第 6 个字节分别存储蓝色、绿色、红色组成信息，以此类推。
       
       numpy array具有ndim、shape和size属性，可以用于获取数组的维度信息。 
             1. ndim属性返回数组的维度数量。      (看数据，几层[]就是几个维度)
             2. shape属性返回数组每个维度的大小。  
             3. size属性返回数组的总元素数量。    (numpy array的索引从0开始)
                如： arr = np.array([[1, 2, 3], [4, 5, 6]])
                    print(arr.ndim)  # 输出：2    
                    print(arr.shape)  # 输出：(2, 3) -> (行 , 列)
                    print(arr.size)  # 输出：6
             reshape()函数用于修改数组的形状，返回一个修改后的数组。    
             flatten()方法用于将多维数组扁平化为一维数组,ravel()方法也可以用于将多维数组扁平化为一维数组。
              
       ndarray.shape (512,512,3)   ndarray.dtype = unit8
       [  
         000(ndarray=(512,3)) =  [  [125 137 226],  [125 137 226],  [133 137 223]...
         001(ndarray=(512,3)) =  [  [123 220 227],  [123 140 117],  [133 137 223]...
         ...                  =  ...        
         512(ndarray=(512,3)) =  [  [57   22  82],  [125 137 226],  [133 137 223]... 
'''
h, w = img.shape[:2]  # 获取图片的high和wide
'''
 Python 中常见的切片操作，切片操作的语法是 sequence[start:end:step]，其中：
    sequence 是要提取元素的序列（例如字符串、列表、元组等）
    start 是提取的起始位置（包含），默认为 0
    end 是提取的结束位置（不包含），默认为序列长度
    step 是提取的步长，默认为 1
    当 start 和 end 都不指定时，[:2] 表示从序列的第一个元素开始，提取两个元素。
'''
img_gray = np.zeros([h, w], img.dtype)  # 生成和当前图片大小([h,w])、类型(dtype)一样的单通道(zeros,全0矩阵)图片
'''
np.zeros是NumPy库中的一个函数，用于创建一个指定形状（shape）和数据类型（dtype）的全零数组。
numpy.zeros(shape, dtype=float, order='C')
     shape: 这是一个必需参数，指定了数组的维度。例如，shape=3 创建一个长度为 3 的一维数组，而 shape=(2,3) 则创建一个 2x3 的二维数组。
     dtype: 这个参数允许用户指定数组中元素的数据类型。常见类型包括 numpy.int32, numpy.float64 等。如果不指定，NumPy 默认使用 float64 类型。img.dtype='unit8'
     order: 不常用，允许高级用户优化他们的数组布局，以便更高效地进行特定的数学运算。可选参数，可以是’C’（按行排列）或’F’（按列排列）。大多数情况下，默认的 'C' 顺序就足够了。
'''
for i in range(h):
    for j in range(w):
        # 取出high和wide中的BGR坐标     m为彩色图片中 当前像素点的值 [ 125 137 226]
        m = img[i, j]
        # 将BGR坐标转化成gray坐标并赋值给新图像  img_gray[i, j]为灰度图中当前像素点的值 126
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
        '''
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                r, g, b = img[i, j, :]
                # 最大值灰度化
                max_gray[i, j] = max(r, g, b)
                # 平均值灰度化
                ave_gray[i, j] = (r + g + b) / 3
                # 加权平均值灰度化
                weight_gray[i, j] = 0.3 * r + 0.59 * g + 0.11 * b
        '''
print(m)
print(img_gray[i, j])
'''
ndarray.shape = (512,512)   ndarray.dtype = unit8
[  000(ndarray(512,))  [162 162 162 ... 169 155 128]
   512(ndarray(512,))  [ 43  43  54 ... 103 105 108] ]
'''
print("image show gray[i,j]: %s" % img_gray)
cv2.imshow("image show gray", img_gray)
""""
waitkey控制着imshow的持续时间，当imshow之后不跟waitkey时，相当于没有给imshow提供时间展示图像，所以只有一个空窗口一闪而过。
添加了waitkey后，哪怕仅仅是cv2.waitkey(1),也能截取到一帧的图像。所以cv2.imshow后边是必须要跟cv2.waitkey的。
"""
cv2.waitKey(10000)  # 显示10s

plt.subplot(221)  # 使用plt.subplot来创建小图,plt.subplot(221)表示将整个图像窗口分为2行2列,当前位置为1
img = plt.imread("D:\cv_workspace\picture\lenna.png")
'''
返回值数据类型取决于图像的格式和读取时的设置
ndarray.shape = (512, 512, 3)    ndarray.dtype = float32
          [[[0.8862745  0.5372549  0.49019608]
          [0.8862745  0.5372549  0.49019608]
          [0.8745098  0.5372549  0.52156866]
'''
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
'''
rgb2gray是matlab内部一种处理图像的函数，通过消除图像色调和饱和度信息同时保留亮度实现将RGB图像或彩色图转换为灰度图像，即灰度化处理的功能，
         调用这个功能的格式是I = rgb2gray(RGB)，意思是将真彩色图像RGB转换为灰度强度图像I 。
'''
img_gray = rgb2gray(img)
'''
ndarray.shape = (512, 512)   ndarray.dtype = float32
    [[0.60802865 0.60802865 0.60779065 ... 0.6413741  0.57998234 0.46985728]
     [0.60802865 0.60802865 0.60779065 ... 0.6413741  0.57998234 0.46985728]
     [0.60802865 0.60802865 0.60779065 ... 0.6413741  0.57998234 0.46985728]
     ...
'''

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.cvtColor() 函数是 OpenCV 中用于图像颜色空间转换的函数。它允许你将图像从一个色彩空间转换为另一个色彩空间,可以使用这个函数来实现不同色彩空间之间的转换。
#     cv2.cvtColor(src, code[, dst[, dstCn]])
#             src：输入图像，可以是 NumPy 数组或 OpenCV 中的 Mat 对象。
#             code：颜色空间转换代码，表示目标色彩空间。可以使用 OpenCV 中的 cv2.COLOR_* 常量来指定，如 cv2.COLOR_BGR2GRAY 表示将 BGR 彩色图像转换为灰度图像。
#             dst：可选参数，输出图像，可以是 NumPy 数组或 Mat 对象。如果未提供，将会创建一个新的图像来保存转换后的结果。
#             dstCn：可选参数，目标图像的通道数。默认值为 0，表示与输入图像通道数保持一致。

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
'''
ndarray.shape = (512, 512)   ndarray.dtype = float32
    [[1 1 1 ... 1 1 0]
     [1 1 1 ... 1 1 0]
     [1 1 1 ... 1 1 0]
     ...
'''
print(img_gray.shape, img_gray.dtype)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

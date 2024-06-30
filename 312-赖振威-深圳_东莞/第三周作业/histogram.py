import cv2  # 导入OpenCV库
import numpy as np  # 导入NumPy库
from matplotlib import pyplot as plt  # 从matplotlib库中导入pyplot模块

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''

# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)  # 读取彩色图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像
cv2.imshow("image_gray", gray)  # 显示灰度图像

# # 灰度图像的直方图，方法一
# '''
# gray.ravel()：这部分代码将灰度图像转换为一个一维数组/矩阵，即将二维的灰度图像展平为一维数组。这样做是为了方便后续直方图的统计。
# plt.hist(...)：这是用来绘制直方图的函数，其中第一个参数是数据，即灰度图像的一维数组；第二个参数是指定直方图的 bin 的个数，这里设定为 256，表示将灰度值分成 256 个 bin 进行统计。
# 所以，整段代码的作用就是将灰度图像展平为一维数组，并根据灰度值绘制出直方图，以展示不同灰度值的像素在图像中的分布情况。通过直方图可以更直观地了解图像的亮度分布情况。
# '''
# plt.figure()  # 创建一个新的图像
# plt.hist(gray.ravel(), 256)  # 绘制灰度图像的直方图
# plt.show()  # 显示直方图图像

# # ----------------------------------------------------------------------------------------------------------------------
# # 灰度图像的直方图, 方法二
# '''
# 这段代码是使用 OpenCV（cv2）库中的 calcHist 函数来计算灰度图像的直方图。让我解释一下：
# cv2.calcHist([gray],[0],None,[256],[0,256])：这部分代码调用了 calcHist 函数来计算直方图。
#     第一个参数 [gray] 表示要计算直方图的图像，这里是灰度图像 gray。
#     第二个参数 [0] 是指定通道索引，对于灰度图像只有一个通道，所以为 0。
#     第三个参数 None 表示不使用掩模（mask），如果需要可以传入一个与输入图像相同尺寸的掩模进行计算。
#     第四个参数 [256] 是指定直方图的 bin 的个数，这里设定为 256，表示将灰度值分成 256 个 bin 进行统计。
#     第五个参数 [0,256] 是指定像素值的范围，这里表示灰度值的范围为 0 到 255。
# 所以，这行代码的作用是利用 calcHist 函数计算灰度图像 gray 的直方图，得到一个包含了灰度值统计信息的直方图数组 hist。这个直方图可以用来分析图像的像素分布情况。
# '''
# hist = cv2.calcHist([gray],[0],None,[256],[0,256])  # 计算灰度图像的直方图
# plt.figure()  # 创建一个新的图像
# plt.title("Grayscale Histogram")  # 设置标题
# plt.xlabel("Bins")  # 设置X轴标签
# plt.ylabel("# of Pixels")  # 设置Y轴标签
# plt.plot(hist)  # 绘制直方图  其实我觉得更像线型图
# plt.xlim([0,256])  # 设置x坐标轴范围
# plt.show()  # 显示直方图图像

# ----------------------------------------------------------------------------------------------------------------------
# 彩色图像直方图
image = cv2.imread("lenna.png")  # 读取彩色图像
cv2.imshow("Original",image)  # 显示原始彩色图像
# cv2.waitKey(0)

chans = cv2.split(image)  # 将彩色图像拆分成三个通道
colors = ("b","g","r")  # 定义颜色通道顺序
plt.figure()
plt.title("Flattened Color Histogram")  # 设置标题
plt.xlabel("Bins")  # 设置X轴标签
plt.ylabel("# of Pixels")  # 设置Y轴标签

for (chan,color) in zip(chans,colors):  # 遍历bgr每个颜色通道并bgr分类来画线型图
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])  # 计算每个颜色通道的直方图
    plt.plot(hist,color = color)  # 绘制直方图
    plt.xlim([0,256])  # 设置x坐标轴范围   不同的bgr程度的数量
plt.show()  # 显示直方图图像

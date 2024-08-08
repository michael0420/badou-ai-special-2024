'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('C:/Users/86188/Pictures/lenna.png', 0)
print (img.shape)

#获取图像高度、宽度
rows, cols = img.shape[:]

#图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()


"""
使用 OpenCV 库的 cv2.kmeans 函数对一张灰度图像进行 K-Means 聚类分析。下面是代码的简化和解释：

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
# 加载图像并转化为灰度图 
img = cv2.imread('lenna.png', 0) 
print(img.shape)  # 打印图像尺寸 
 
# 重组数据为一维数组 
data = img.reshape((-1, 1))  # 重要的是这里要保持数据维度为 np.float32 类型 
 
# 设置 K-Means 参数 
K = 4  # 设置类簇数量为4 
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 设置迭代停止条件 
flags = cv2.KMEANS_RANDOM_CENTERS  # 随机初始化中心点 
 
# 运行 K-Means 算法 
compactness, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)
 
# 显示结果 
dst = labels.reshape((img.shape[0], img.shape[1]))
plt.figure(figsize=(10, 5))  # 设置图像大小 
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(dst, cmap='gray'), plt.title('Clustered Image')
plt.xticks([]), plt.yticks([])
plt.show()

在这个例子中：

    cv2.imread('lenna.png', 0) 加载图像并转化为灰度图。
    data = img.reshape((-1, 1)) 将图像二维像素转换为一维，注意这里保持数据维度为 np.float32 类型。
    criteria 设置迭代停止的条件，包括最大迭代次数和误差容忍度。
    flags = cv2.KMEANS_RANDOM_CENTERS 指定随机初始化中心点。
    cv2.kmeans(data, K, None, criteria, 10, flags) 运行 K-Means 算法。
    dst = labels.reshape((img.shape[0], img.shape[1])) 将标签结果转换回与原始图像相同的形状。
    最后，使用 matplotlib 库显示原始图像和聚类后的结果。

请注意，K-Means 算法的结果可能会因每次运行的随机初始化而有所不同。此外，这个算法对于不同的图像和应用场景可能需要调整参数，
比如 K 值的选择和其他设置。
"""
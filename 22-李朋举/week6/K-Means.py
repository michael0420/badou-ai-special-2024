# coding: utf-8



import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('D:\cv_workspace\picture\lenna.png', 0)
print(img.shape)

# 获取图像高度、宽度
rows, cols = img.shape[:]

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
# np.float32是numpy库中一个数据类型，表示单精度浮点数，占用 4 个字节, 了确保数据对象是单精度浮点数类型
data = np.float32(data)  # ndarrary = (262144,1)

# 停止条件 (type,max_iter,epsilon)
'''
使用了 OpenCV 库中的TermCriteria结构体来定义停止条件。这个结构体包含了三个参数：
type：表示停止条件的类型。在你的例子中，使用了cv2.TERM_CRITERIA_EPS，这意味着当误差（error）达到某个阈值（epsilon）时，迭代将停止。
max_iter：表示最大迭代次数。在你的例子中，设置为 10 次迭代。
epsilon：表示误差阈值。在你的例子中，设置为 1.0。
当迭代次数达到max_iter，或者误差小于等于epsilon时，迭代将停止。
'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # (3, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS  # flags:0
'''
KMEANS_RANDOM_CENTERS表示初始质心是随机选择的，而不是使用默认的方式（即第一个样本作为第一个质心，然后计算其他质心与第一个质心的距离，选择距离最远的样本作为下一个质心）。
随机选择初始质心可以提高算法的鲁棒性，尤其在数据集具有不同分布或存在噪声的情况下。
'''

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
'''
对输入数据进行 K-Means 聚类，并返回簇的紧致性度量、标签和中心
在OpenCV中，Kmeans()函数原型如下所示： retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
      入参：  
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
        
        这段 Python 代码使用 OpenCV（Open Computer Vision）库中的`cv2.kmeans()`函数执行 K-Means 聚类算法。它将输入数据`data`分为`4`个簇，并返回三个参数：`compactness`，`labels`和`centers`。
      返参：
        1. `compactness`：这是一个返回值，它是每个簇的紧致性的度量。紧致性是一种评估簇质量的方法，它表示簇内数据的凝聚力或紧密程度。紧致性值越小，说明簇内数据越紧密，簇的质量越高。
               35875000.75155999 
        2. `labels`：整数数组，长度与输入数据`data`的长度相同。它为输入数据中的每个元素分配了一个唯一的簇标签。这些标签从`0`到`3`，分别表示`4`个簇。
              ndarrary = (262144,1)   
               [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], 
                [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [3], [3],
                [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3],
                [3], [3], [3], [3] ...]
        3. `centers`：表示每个簇的中心，这是一个包含`4`个元素的数组，每个元素是一个包含`data`维度的数组。中心是通过计算簇内元素的平均值得到的。
               [[195.22253], [152.49554], [ 64.9078 ], [112.68654]]
'''

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))
'''
(512,512)  [[1 1 1 ... 0 1 1], [1 1 1 ... 0 1 1], [1 1 1 ... 0 1 1], ..., [3 3 3 ... 1 1 1], [3 3 3 ... 1 1 1], [3 3 3 ... 1 1 1] ...]
'''
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 字体设置为SimHei，这是一种中文字体

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    # 在一个包含 2 行 1 列的图形中创建第i + 1个子图，子图的索引从左上角开始，从左到右，从上到并在该子图中显示图像images[i]，并将其显示为灰度
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]),  # 表示不使用任何刻度标签，因此 x 轴将没有任何标记。这通常用于创建一个空白的 x 轴，或者在 x 轴上显示一些自定义的标记，而不是默认的数字标签。
    plt.yticks([])
plt.show()

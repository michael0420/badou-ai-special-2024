import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png",0)
print(img.shape)


rows, cols = img.shape[:]

data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件(type, max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            10)

#设置标签 在每次尝试中选择随机的初始中心
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类
"""
kmeans( data,              -用于聚类的数据
        K,                 -用来分割集合的集群数
        bestLabels,        -用于存储每个样本的聚类索引
        criteria,          -算法终止条件
                            当达到最大循环数目或者指定的精度阈值时，算法停止继续分类迭代计算
                            该参数由3个子参数构成，分别为type、max_iter和eps
                            type表示终止的类型，可以是三种情况
                                cv2.TERM_CRITERIA_EPS：精度满足eps时，停止迭代
                                cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过阈值max_iter时，停止迭代
                                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER：上述两个条件中的任意一个满足时，停止迭代
                            max_iter：最大迭代次数。
                            eps：精确度的阈值。
        attempts,          -在具体实现时，为了获得最佳分类效果，可能需要使用不同的初始分类值进行多次尝试
                            指定attempts的值，可以让算法使用不同的初始值进行多次（attempts次）尝试。
        flags,              KMEANS_RANDOM_CENTERS - 在每次尝试中选择随机的初始中心。
                            KMEANS_PP_CENTERS - 使用Arthur和Vassilvitskii进行的kmeans ++中心初始化。
                            KMEANS_USE_INITIAL_LABELS - 使用用户输入的数据作为第一次分类中心点；如果算法需要尝试多次（attempts 值大于1时），后续尝试都是使用随机值或者半随机值作为第一次分类中心点。




)


返回值：
    compactness 聚类的紧凑度（样本到其所属聚类中心的距离平方和）
    labels      一个与输入数据大小相同的一维数组，包含每个样本的标记，标记的取值范围为 0 到 K-1
    centers     聚类的中心。如果未提供，则由函数计算。中心矩阵的大小为 K 行（每个簇一个中心），每行数据的列数与输入数据的列数相同
"""

compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

print(compactness)
print(labels)
print(centers)
#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()



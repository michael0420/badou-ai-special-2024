# coding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

#读取图片
img = cv2.imread('lenna.png')
print (img.shape)
# 转一维数组，保留3通道
data = img.reshape((-1, 3))
#转float32
data = np.float32(data)
# 设置 K-Means 参数
K = 2
#算法终止条件
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#设置标签 随机点
flags = cv2.KMEANS_RANDOM_CENTERS
# 重复试验kmeans算法的次数
attempts = 10
#K-Means聚类 聚集成2类
kList = [2**x for x in range(6) if x > 0]
print("kList", kList)
imgs = []
for k in kList:
    retval, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    # 将中心点转换为整型
    centers = np.uint8(centers)

    # 根据标签将图像重新映射到新的颜色空间
    segmented_image = centers[labels.flatten()]

    # 将图像重新整形为原始形状
    dst_image = segmented_image.reshape((img.shape))
    # 图像转换为RGB显示
    dst_image = cv2.cvtColor(dst_image, cv2.COLOR_BGR2RGB)
    imgs.append(dst_image)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgs.append(img)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
titles = [u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64', u'原始图像']

for i in range(len(imgs)):
    plt.subplot(2,3,i+1), plt.imshow(imgs[i], 'gray')
    plt.title(titles[i])
    # 设置隐藏x,y轴标签刻度
    plt.xticks([]), plt.yticks([])

plt.show()
"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/5/16 10:12
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../lenna.png")
# 将每个像素展开成包含三个值(BGR)的行, 并且类型转为float32
data_X = img.reshape((-1, 3))
data = np.float32(data_X)
# 聚类数为4, 分为四种颜色
K = 4
bestLabels = None
# 设置k-means算法的终止条件：迭代10次或者当误差小于1.0时停止
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# attempts: 尝试k-means算法次数
attempts = 10
# 设置标签, 随机获取质心
flags = cv2.KMEANS_RANDOM_CENTERS
# retval: 紧凑型度量, label: 每个数据点的聚类标签, centers: 聚类中心
retval, label, centers = cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)
# 图像转换回uint8二维类型
centers = np.uint8(centers)  # 将聚类中心转换回uint8类型，以便后续用于图像重建(聚类中心即为要显示的颜色值)。
res = centers[label.flatten()]  # 根据聚类标签选择相应的聚类中心颜色值。
dst_img = res.reshape((img.shape))  # 将一维数组重新塑形为原始图像的形状。
# 图像转换为RGB显示
dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
# 显示聚类后的图像
plt.imshow(dst_img)
plt.title(u'聚类图像')
plt.xticks([]), plt.yticks([])  # 去除坐标印记
plt.show()

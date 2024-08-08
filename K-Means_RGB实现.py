import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('C:/Users/86188/Pictures/lenna.png')
print (img.shape)

#图像二维像素转换为一维
data = img.reshape((-1,3))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)

#K-Means聚类 聚集成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)

#K-Means聚类 聚集成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)

#K-Means聚类 聚集成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)

#K-Means聚类 聚集成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

#图像转换回uint8二维类型
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()

"""
使用 OpenCV 库的 cv2.kmeans 函数对一张彩色图像进行 K-Means 聚类分析。下面是代码的简化和解释：

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
# 加载图像 
img = cv2.imread('lenna.png') 
print(img.shape)  # 打印图像尺寸 
 
# 重组数据为一维数组 
data = img.reshape((-1, 3))  # 重要的是这里要保持数据维度为 np.float32 类型 
 
# 设置 K-Means 参数 
K = [2, 4, 8, 16, 64]  # 设置不同的类簇数量 
criteria = (cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 设置迭代停止条件 
flags = cv2.KMEANS_RANDOM_CENTERS  # 随机初始化中心点 
 
# 运行 K-Means 算法并可视化结果 
for k in K:
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    
    # 将结果转换回原始图像形状 
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape(img.shape)
 
    # 将原始图像和聚类结果转换为 RGB 格式以便显示 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
 
    # 可视化图像 
    plt.figure(figsize=(10, 5))  # 设置图像大小 
    plt.subplot(2, 3, i+1), plt.imshow(dst), plt.title('Clustered Image K={}'.format(k))
    plt.xticks([]), plt.yticks([])
 
# 显示所有图像 
plt.show()

在这个例子中：

    cv2.imread('lenna.png') 加载图像。
    data = img.reshape((-1, 3)) 将图像二维像素转换为一维，注意这里保持数据维度为 np.float32 类型。
    criteria 设置迭代停止的条件。
    flags = cv2.KMEANS_RANDOM_CENTERS 指定随机初始化中心点。
    cv2.kmeans(data, k, None, criteria, 10, flags) 运行 K-Means 算法。
    dst = res.reshape(img.shape) 将标签结果转换回与原始图像相同的形状。
    plt.subplot(2, 3, i+1), plt.imshow(dst), plt.title('Clustered Image K={}'.format(k)) 在子图中显示聚类结果，
    便于对比不同 K 值的效果。

请注意，选择合适的 K 值对于聚类结果来说非常重要，过大或过小的 K 值都可能导致聚类效果不佳。此外，
这个算法对于不同的图像和应用场景可能需要调整参数。
"""
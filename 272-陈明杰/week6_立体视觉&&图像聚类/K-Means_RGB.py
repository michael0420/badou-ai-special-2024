import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1、读取图片
img = cv2.imread("lenna.png")
# 转换成一维数组，不能直接用img接收，因为后面的代码中使用了img.shape，曾经踩坑
data = img.reshape(-1, 3)
# 转化成浮点数数组，因为kmeans函数必须要求是浮点数类型
data = np.float32(data)

# 2、图像聚类
# (1)停止条件
# 第一个元素是停止条件的类型。这里，你使用了cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS，
# 这意味着算法将在达到最大迭代次数或两次迭代之间的变化小于指定阈值时停止。
# 第二个元素是最大迭代次数，这里设置为10。
# 第三个元素是阈值（epsilon），这里设置为0.2。
#
# 以下停止条件代表：
# 如果算法迭代了10次（或更少），则停止。
# 或者，如果连续两次迭代之间的变化小于0.2（在某种度量下，
# 通常是质心之间的距离或数据点到质心的距离的平方和），则停止。
Criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10, 0.2)
# cv2.KMEANS_RANDOM_CENTERS 是 flags 参数的一个选项，用于指定初始质心（或称为聚类中心）
# 的选择方法。当 flags 被设置为 cv2.KMEANS_RANDOM_CENTERS 时，初始质心是从数据集中随机选择的。
# 这是使用 cv2.KMeans 函数时 flags 参数的一个常见设置。其他可能的选项包括：
# cv2.KMEANS_PP_CENTERS：使用K-means++算法来选择初始质心。K-means++算法通常能提供更好的
# 初始质心，从而加快算法的收敛速度并可能得到更好的聚类结果。
Flags = cv2.KMEANS_RANDOM_CENTERS
# Flags = cv2.KMEANS_PP_CENTERS

# (2)图像聚类:
# compactness是一个浮点数字表示所有点到其相应聚类中心的距离的平方和（即误差的总和）
# labels是一个一维数组，表示每一个像素点对应属于哪一个聚类集合，例如1/2/3分别代表该像素点属于聚类1/2/3,
# 指示该点属于哪个聚类（从 0 到 K-1，其中 K 是聚类的数量）
# centers是一个 NumPy 数组，大小为 (K, D)，其中 K 是聚类的数量，D 是输入数组 img 中
# 每个数据点的维数。这个数组包含了每个聚类的中心点的坐标

# 聚成2类
# criteria中的最大迭代次数和阈值是对于某一次聚类中的，而attempts代表的是该kmeans函数
# 会执行attempts次，每次都取KMEANS_RANDOM_CENTERS随机质心，然后把每次执行后得到的结果
# 进行对比，选择最优的作为最终结果
compactness, labels, centers2 = cv2.kmeans(data, 2, None, criteria=Criteria, attempts=10, flags=Flags)
centers2 = np.uint8(centers2)

# labels 是一个一维或二维数组，其中包含了每个样本所属的聚类标签（即聚类索引）。
# .flatten() 是一个NumPy数组的方法，用于将多维数组转换为一维数组。但在这里，
# 如果labels本身就是一维的（如在cv2.kmeans函数的典型用法中），.flatten()将不
# 会改变数组，但这样做通常是为了确保无论输入是什么形状，都能得到一个一维数组。

# 这是一个高级索引操作，它使用labels.flatten()数组中的值作为索引来从centers2数组中选取元素。
# 具体来说，对于labels.flatten()中的每个值（即每个样本的聚类标签），它都会从centers2数组中
# 选取对应行的质心。
# 结果是一个形状与原始数据相同（或至少一维形状相同）的数组，但每个元素现在都是一个质心，表示原
# 始数据中对应样本所属的聚类。
# 最后，res将包含每个样本所属的聚类的质心。这在某些应用中可能很有用，例如当你想要用聚类的质心
# 来代替原始数据集中的样本时。
# 这样操作之后，res中的每一个元素的像素值要么是聚类质心一的像素值，要么是聚类质心二的像素值，
# 这样就实现了把原图分成两种类型了
res = centers2[labels.flatten()]

# 如果data是图像的像素数据，那么centers2中的值就是像素值。它们代表了每个
# 聚类在颜色空间中的中心点，可以用于各种应用，如图像分割、颜色量化等。
dst2 = res.reshape(img.shape)
# 因为cv2读取进来的图像是BGR格式的，所以要转变回RGB格式再显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)

# 聚类成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria=Criteria, attempts=10, flags=Flags)
centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape(img.shape)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)

# 聚类成8类
compactness, labels8, centers8 = cv2.kmeans(data, 4, None, criteria=Criteria, attempts=10, flags=Flags)
centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape(img.shape)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)

# 聚类成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, Criteria, 10, Flags)
centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape(img.shape)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)

# 聚类成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, Criteria, attempts=10, flags=Flags)
centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape(img.shape)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = ['原始图像', '聚类图像 K=2', '聚类图像 K=4',
          '聚类图像 K=8', '聚类图像 K=16', '聚类图像 K=64']
images = [img, dst2, dst4, dst8, dst16, dst64]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

img = cv2.imread('lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 转换格式，转换数据类型
high, width = img.shape[:2]
src = img.reshape((high*width, 3))
src = np.float32(src)
# 停止条件
Criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 10, 1.0)
# 随机质心
Flags = cv2.KMEANS_RANDOM_CENTERS

# 聚成2类
res2, labels2, centers2 = cv2.kmeans(src, 2, None, Criteria, 10, Flags)
# 这几行不能漏掉，否则就reshape失败了
print(labels2.shape)
centers2 = np.uint8(centers2)
res = centers2[labels2]
print(res.shape)
dst2 = res.reshape(img.shape)
print(dst2.shape)

# 聚成4类
res4, labels4, centers4 = cv2.kmeans(src, 4, None, Criteria, 10, Flags)
centers4 = np.uint8(centers4)
res = centers4[labels4]
dst4 = res.reshape(img.shape)
# 聚成8类
res8, labels8, centers8 = cv2.kmeans(src, 8, None, Criteria, 10, Flags)
centers8 = np.uint8(centers8)
res = centers8[labels8]
dst8 = res.reshape(img.shape)
# 聚成16类
res16, labels16, centers16 = cv2.kmeans(src, 16, None, Criteria, 10, Flags)
centers16 = np.uint8(centers16)
res = centers16[labels16]
dst16 = res.reshape(img.shape)
# 聚成64类
res64, labels64, centers64 = cv2.kmeans(src, 64, None, Criteria, 10, Flags)
centers64 = np.uint8(centers64)
res = centers64[labels64]
dst64 = res.reshape(img.shape)

# 画图
images = [img, dst2, dst4, dst8, dst16, dst64]
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = ['原始图像', '聚2类', '聚4类', '聚8类', '聚16类', '聚64类']
plt.figure()
for i in range(6):
    plt.subplot(2, 3, 1 + i)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()

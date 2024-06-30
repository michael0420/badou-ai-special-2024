import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

img_gray = cv2.imread('lenna.png', 0)
img = cv2.imread('lenna.png')

rows, cols = img_gray.shape

#print(img)
img_tmp = img.copy()
data = img_tmp.reshape((-1,3))
data = np.float32(data)


data_gray = img_gray.reshape((rows * cols, 1))
data_gray = np.float32(data_gray)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_PP_CENTERS

compactness, labels, centers = cv2.kmeans(data_gray, 5, None, criteria, 10, flags)
compactness, labels1, centers1 = cv2.kmeans(data, 6, None, criteria, 10, flags)
dst = labels.reshape((img_gray.shape[0], img_gray.shape[1]))
centers1 = np.uint8(centers1)
res = centers1[labels1.flatten()]
dst1 = res.reshape((img.shape))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst1 = cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB)
plt.rcParams['font.sans-serif']=['SimHei']


src_arr, labels=load_iris(return_X_y=True) #加载数据
print(src_arr.shape)
des_arr = src_arr.reshape((-1,2))
print(des_arr.shape)
clf = KMeans(n_clusters=3)
y_pred = clf.fit_predict(des_arr)
x = [n[0] for n in des_arr]

y = [n[1] for n in des_arr]

scatter = plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(handles=scatter.legend_elements()[0], labels=["A", "B", "C"])
# 显示图形
plt.show()
titles = [u'原始灰度图', u'聚类灰度图', u'原始彩色图', u'聚类彩色图']
images = [img_gray, dst, img, dst1]
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks()
    plt.yticks()
plt.show()
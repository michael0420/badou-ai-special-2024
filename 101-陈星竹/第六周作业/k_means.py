import cv2
import numpy as np
import matplotlib.pyplot as plt

#灰度读图
img = cv2.imread('lenna.png',0)
print(img.shape)

#图像宽高
h,w = img.shape

#转换成一维 方便运算
data = img.reshape((h * w),1)
data = np.float32(data)

#定义k-means条件
k = 4
beatLabels = None
#迭代终止条件
'''
cv2.TERM_CRITERIA_EPS:误差(精确度)
cv2.TERM_CRITERIA_MAX_ITER:最大迭代次数
cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER:表示满足其一就停止迭代
'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,0.1)

#算法重复次数，最终选择误差最小的结果
attempts = 10

#初始化聚类中心的方法
flags = cv2.KMEANS_RANDOM_CENTERS

#预设的初始聚类中心
center = None

'''
#调用k-means接口
cv2.kmeans 返回三个值：
    ret：紧缩的总和
    label：每个数据点的标签，表示它属于哪个簇。
    center：每个簇的中心。
'''
ret,labels,centers = cv2.kmeans(data,k,beatLabels,criteria,attempts,flags)

#生成最终图像
dst = labels.reshape((h,w))

#将默认字体设置为SimHei（黑体），以便正常显示中文标签。
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像',u'聚类图像']
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([]) #去掉坐标轴刻度
plt.show()

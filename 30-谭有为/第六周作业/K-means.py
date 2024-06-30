import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans

#篮球运动员的聚类
basketball_data=[[0.0888, 0.5885],
[0.1399, 0.8291],
 [0.0747, 0.4974],
 [0.0983, 0.5772],
 [0.1276, 0.5703],
 [0.1671, 0.5835],
 [0.1306, 0.5276],
 [0.1061, 0.5523],
 [0.2446, 0.4007],
 [0.1670, 0.4770],
 [0.2485, 0.4313],
 [0.1227, 0.4909],
 [0.1240, 0.5668],
 [0.1461, 0.5113],
 [0.2315, 0.3788],
 [0.0494, 0.5590],
 [0.1107, 0.4799],
 [0.1121, 0.5735],
 [0.1007, 0.6318],
 [0.2567, 0.4326],
 [0.1956, 0.4280]
]
print(basketball_data)
k=KMeans(n_clusters=3)   #设置聚类的簇数
kmeans_result=k.fit_predict(basketball_data) #引用sklearn.cluster库中的kmeans接口
print(kmeans_result)
#得到聚类结果后画图
x=[i[0] for i in basketball_data]
y=[i[1] for i in basketball_data]
plt.scatter(x,y,c=kmeans_result,marker='*')
plt.title('Kmeans-Basketball Data')
plt.xlabel('assists_per_minute')
plt.ylabel('points_per_minute')
plt.legend(loc='best',title='kmeans')  #添加图例
plt.show()


#灰色图像的聚类效果
img=cv2.imread('F:/PNG/lenna.png',0) #读取灰度图
heiht,width=img.shape[:2]
data=img.reshape((heiht*width),1)  #二维数据转换为一维数据
data=np.float32(data)  #转换数据类型，kmeans的输入数据最好为mp.float32格式
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1) #设置终止条件,这里表示当达到聚类10次或者误差小于1时，终止
flags=cv2.KMEANS_RANDOM_CENTERS  #随机初始中心
compactness,Labels,centers=cv2.kmeans(data,4,None,criteria,10,flags)
#kmeans聚类,返回数据中，compactness表示紧密度，即每个点到对应聚类中心距离的平方和，labels表示 标志数组  centers表示 聚类中心组成的数组
print(Labels)
Labels=np.uint8(Labels)/3   #转换数据类型，除以3转换为0-1之间的浮点数
kmeans_img=Labels.reshape((heiht,width))

#cv显示
cv2.imshow('img',img)
cv2.imshow('kmeans img',kmeans_img)


'''
#plt显示
plt.subplot(211)
plt.imshow(img,'gray')
plt.subplot(212)
plt.imshow(kmeans_img,'gray')
plt.show()
'''

#彩色图像的聚类效果
img1=cv2.imread('F:/PNG/lenna.png')
print(img1)
data1=img1.reshape(-1,3)  #reshape（-1，3）原数组转换成n行3列的矩阵
print(data1)
data1=np.float32(data1)
compactness,Labels1,centers1=cv2.kmeans(data1,4,None,criteria,10,flags) #kmeans聚类
centers1=np.uint8(centers1)
Labels12=Labels1.flatten()  #二维变一维
print('Labels12',Labels12)
print('centers1',centers1)
res=centers1[Labels12]  #Labels中的标签用中心点的值代替
print('res',res)
kmeans_img1=res.reshape((img1.shape))
cv2.imshow('img1',img1)
cv2.imshow('kmeans img1',kmeans_img1)
cv2.waitKey()
cv2.destroyAllWindows()




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





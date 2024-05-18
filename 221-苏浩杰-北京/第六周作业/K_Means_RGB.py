import  cv2
import numpy as np
import matplotlib.pyplot as plt



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

img=cv2.imread("../lenna.png")
x,y,channel=img.shape
data=img.reshape(-1,channel)
data=np.float32(data)

#停止条件
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1)

#聚类中心
flag=cv2.KMEANS_RANDOM_CENTERS
#迭代次数
attempts=10

#K-Means聚类 聚集成2类
compactness, labels2, centers2 =cv2.kmeans(data,2,None,criteria,attempts,flag)

compactness, labels4, centers4=cv2.kmeans(data,4,None,criteria,attempts,flag)

compactness, labels8, centers8=cv2.kmeans(data,8,None,criteria,attempts,flag)

compactness, labels16, centers16=cv2.kmeans(data,16,None,criteria,attempts,flag)

compactness, labels32, centers32=cv2.kmeans(data,32,None,criteria,attempts,flag)


#转换成uint二维类型
def tran_uint8(center,labels):
    center=np.uint8(center)
    labels=labels.flatten()
    res=center[labels]
    res=res.reshape(x,y,channel)
    return res

dst2=tran_uint8(centers2,labels2)
dst4=tran_uint8(centers4,labels4)
dst8=tran_uint8(centers8,labels8)
dst16=tran_uint8(centers16,labels16)
dst32=tran_uint8(centers32,labels32)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
img4=cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
img8=cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
img16=cv2.cvtColor(dst16 ,cv2.COLOR_BGR2RGB)
img32=cv2.cvtColor(dst32,cv2.COLOR_BGR2RGB)

titles=['原图','聚2类图像','聚4类图像','聚8类图像','聚16类图像','聚32类图像']
images=[img,img2,img4,img8,img16,img32]
for i in range(len(images)):
    plt.subplot(2,3,i+1),plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
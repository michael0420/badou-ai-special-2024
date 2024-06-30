import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data():
    """
    加载数据集
    """
    X = np.array([
        [0.0888, 0.5885],
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
    ])
    return X

def kmeans_clustering(X, n_clusters):
    """
    使用K-Means算法进行聚类
    """
    clf = KMeans(n_clusters=n_clusters)
    y_pred = clf.fit_predict(X)
    return y_pred

def plot_data(X, y_pred):
    """
    可视化数据
    """
    x = [n[0] for n in X]
    y = [n[1] for n in X]
    plt.scatter(x, y, c=y_pred, marker='x')
    plt.title("Kmeans-Basketball Data")
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(["A","B","C"])
    plt.show()

def main():
    # 加载数据
    X = load_data()
    # 使用K-Means算法进行聚类
    y_pred = kmeans_clustering(X, 3)
    # 可视化数据
    plot_data(X, y_pred)

if __name__ == "__main__":
    main()


#计算欧式距离
def eucDistance(vec1,vec2):
    return sqrt(sum(pow(vec2-vec1,2)))

#初始聚类中心选择
def initCentroids(dataSet,k):
    numSamples,dim = dataSet.shape
    centroids = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0,numSamples))
        centroids[i,:] = dataSet[index,:]
    return centroids

#K-means聚类算法，迭代
def kmeanss(dataSet,k):
    numSamples = dataSet.shape[0]
    clusterAssement = np.mat(np.zeros((numSamples,2)))
    clusterChanged = True
    #  初始化聚类中心
    centroids = initCentroids(dataSet,k)
    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 找到哪个与哪个中心最近
            for j in range(k):
                distance = eucDistance(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist = distance
                    minIndex = j
              # 更新簇
            clusterAssement[i,:] = minIndex,minDist**2
            if clusterAssement[i,0]!=minIndex:
                clusterChanged = True
         # 坐标均值更新簇中心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssement[:0].A==j)[0]]
            centroids[j,:] = np.mean(pointsInCluster,axis=0)
    print('Congratulations,cluster complete!')
    return centroids,clusterAssement

#聚类结果显示
def showCluster(dataSet,k,centroids,clusterAssement):
    numSamples,dim = dataSet.shape
    mark = ['or','ob','og','ok','^r','+r','<r','pr']
    if k>len(mark):
        print('Sorry!')
        return 1
    for i in np.xrange(numSamples):
        markIndex = int(clusterAssement[i,0])
        plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
    plt.show()



#导入模块
## 使用源码方式
# %load kmeans.py
from math import sqrt
#计算欧式距离
def eucDistance(vec1,vec2):
    return sqrt(sum(pow(vec2-vec1,2)))

#初始聚类中心选择
def initCentroids(dataSet,k):
    numSamples,dim = dataSet.shape
    centroids = np.zeros((k,dim))
    for i in range(k):
        index = int(np.random.uniform(0,numSamples))
        centroids[i,:] = dataSet[index,:]
    return centroids

#K-means聚类算法，迭代
def kmeanss(dataSet,k):
    numSamples = dataSet.shape[0]
    clusterAssement = np.mat(np.zeros((numSamples,2)))
    clusterChanged = True
    #  初始化聚类中心
    centroids = initCentroids(dataSet,k)
    while clusterChanged:
        clusterChanged = False
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            # 找到哪个与哪个中心最近
            for j in range(k):
                distance = eucDistance(centroids[j,:],dataSet[i,:])
                if distance<minDist:
                    minDist = distance
                    minIndex = j
              # 更新簇
            clusterAssement[i,:] = minIndex,minDist**2
            if clusterAssement[i,0]!=minIndex:
                clusterChanged = True
         # 坐标均值更新簇中心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssement[:0].A==j)[0]]
            centroids[j,:] = np.mean(pointsInCluster,axis=0)
    print('Congratulations,cluster complete!')
    return centroids,clusterAssement




#导入模块
import  numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#从文件加载数据集
dataSet=[]
fileIn = open('./testSet.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    dataSet.append([float(lineArr[0]),float(lineArr[1])])

#调用k-means进行数据聚类
dataSet = np.mat(dataSet)
k = 4
centroids,clusterAssement =kmeanss(dataSet,k)

#显示结果
showCluster(dataSet,centroids,clusterAssement)
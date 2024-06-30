'''
@author linrenkang
层次聚类和密度聚类
'''


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.datasets import load_iris


class MyData:
    # 源数据
    srcD = None

    # 绘图
    distanceList = None
    levelCluster = None

    # 簇分类（用于密度函数）
    clusterCnt = 3
    clusters = []

    # 图像颜色坐标
    redLineX, redLineY = [], []
    greenLineX, greenLineY = [], []
    blueLineX, blueLineY = [], []

    def __init__(self, srcD):
        self.srcD = srcD

    # ------------ 私有函数 -------------
    def _Draw(self):
        plt.show()

    def _ShowClusterPic(self):
        plt.scatter(self.clusters[0][:, 0], self.clusters[0][:, 1], c='r', marker='X')
        plt.scatter(self.clusters[1][:, 0], self.clusters[1][:, 1], c='g', marker='.')
        plt.scatter(self.clusters[2][:, 0], self.clusters[2][:, 1], c='b', marker='+')
        plt.show()

    #----------------- 导出接口 -------------
    # 层次聚类
    def levelAlgorithm(self):
        # 计算各个点的距离
        distanceList = linkage(self.srcD, 'ward')
        # 进行层次聚类
        fcluster(distanceList, 4, 'distance')
        # 树状图像处理
        dendrogram(distanceList)
        # 画图
        self._Draw()

    # 密度聚类
    def dbAlgorithm(self):
        # 设置密度算法的参数
        dbscan = DBSCAN(eps=0.4, min_samples=9)
        # 进行密度算法
        dbscan.fit(self.srcD)
        # 整理分类并显示
        self.clusters = [[] for _ in range(self.clusterCnt)]
        self.clusters[0] = self.srcD[dbscan.labels_ == 0]
        self.clusters[1] = self.srcD[dbscan.labels_ == 1]
        self.clusters[2] = self.srcD[dbscan.labels_ == 2]
        self._ShowClusterPic()

'''
以下为层次聚类和密度聚类的应用
'''

if __name__ == "__main__":
    srcD, _ = load_iris(return_X_y=True)
    myData = MyData(srcD)
    myData.levelAlgorithm()
    myData.dbAlgorithm()

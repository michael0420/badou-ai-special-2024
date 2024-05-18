import numpy as np
from sklearn.decomposition import PCA


class PCAMain:
    # 样本矩阵
    src = None
    # 降维目标阶
    target = 0
    # 样本矩阵的中心化
    centerSrc = None
    # 样本集的协方差矩阵
    covariance = None
    # 样本矩阵的降维转换矩阵
    conversion = None
    # 样本矩阵的降维矩阵
    Z = None
    # 特征值
    featureVal = None
    # 特征向量
    featureVec = None

    def __init__(self, src, target):
        self.src = src
        self.target = target

    # 样本矩阵的中心化
    def getCenter(self):
        self.centerSrc = []
        # 样本集的特征均值
        mean = np.array([np.mean(attr) for attr in self.src.T])
        # 样本集的中心化
        self.centerSrc = self.src - mean

    # 样本矩阵的协方差矩阵
    def getCovariance(self):
        # 样本集的样例总数
        ns = np.shape(self.centerSrc)[0]
        # 样本集的协方差矩阵
        self.covariance = np.dot(self.centerSrc.T, self.centerSrc) / (ns - 1)

    # 样本矩阵的协方差矩阵的特征值和特征向量
    def getFeature(self):
        self.featureVal, self.featureVec = np.linalg.eig(self.covariance)

    # 样本矩阵的降维转换矩阵
    def getConversion(self):
        ind = np.argsort(-1 * self.featureVal)
        # 构建K阶降维的降维转换矩阵U
        UT = [self.featureVec[:, ind[i]] for i in range(self.target)]
        self.conversion = np.transpose(UT)

    def getZ(self):
        self.Z = np.dot(self.src, self.conversion)
        print("样本矩阵的降维矩阵为：", self.Z)


if __name__ == '__main__':
    # 产生随机数据
    randMinVal = 0
    randMaxVal = 100
    randShape = (4, 5)
    X = np.random.randint(randMinVal, randMaxVal, randShape)

    # PCA
    myData = PCAMain(X, 3)

    # 本地实现pca
    myData.getCenter()
    myData.getCovariance()
    myData.getFeature()
    myData.getConversion()
    myData.getZ()

    # 调用接口 PCA
    # 降维到3维
    pca = PCA(n_components=3)
    # 执行
    pca.fit(X)
    # 获取降维后的数据
    pcaData = pca.fit_transform(X)
    print("调用接口降维后的数据", pcaData)
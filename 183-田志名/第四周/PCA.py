import  numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

class pca(object):
    def __init__(self,data,k):
        self.X=data         #样本矩阵 10*3
        self.K=k            #k为降维后的维度数
        self.centerX=[]     #矩阵x的中心化 10*3
        self.C=[]           #协方差矩阵   3*3
        self.Z = []         #样本矩阵X的降维矩阵Z 10*2

        #1:对原始数据零均值化（中心化）10*3
        self.centerX = self.center()
        #2:协方差矩阵的构造,3*3
        self.C=self.cov()
        #3:求取协方差的特征值及特征向量，这些特征向量组成了新的特征空间
        self.Z=self.reductDimension()
        print(self.Z)


    def center(self):
        centX=[]
        mean=np.array([np.mean(i) for i in self.X.T])     #mean为1*3的矩阵，i是一列的数据
        #mean = np.mean(self.X, axis=0)
        centX=self.X-mean                                 #x为10*3，mean广播为10*3，两者相减
        return centX

    def cov(self):
        #协方差矩阵为中心化矩阵x的转置乘以中心化矩阵，再除以样本个数
        return np.dot(self.centerX.T,self.centerX)/self.X.shape[0]

    def reductDimension(self):
        a, b = np.linalg.eig(self.C)                          #求取特征值和特征向量，a是特征值，b是特征向量，都是1*3的
        index=np.argsort(-1*a)                                #np.argsort返回排序完成后的下标，且默认从小到大排，乘以-1改变排序方式
        newmatrix=np.array([b[:,index[i]] for i in range(self.K)]).T  #取出特征值最大的K个对应下标的列，newmatix为3*2，特征数从3变成2
        return np.dot(self.X,newmatrix)                       #10*3，3*2，最后为10*2，从3个特征中提取出2个最重要的


if __name__=='__main__':
    #特征数量为3的样本，通过pca将维成2维
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K=X.shape[1]-1
    result=pca(X,K)

    #简洁版
    #用load_iris的一个数据集来验证，分类任务。数据集是150*4的，特征有4个，数据标签三个类别。现在将其降维成2个，若区分结果依然有效，则pca效果可以
    x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签，分类任务
    pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(x)  # 对原始数据进行降维，保存在reduced_x中，150*2
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()



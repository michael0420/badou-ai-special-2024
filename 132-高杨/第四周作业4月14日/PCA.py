import numpy as np
from sklearn.decomposition import PCA as pca

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris


class PCAorgin:
    def __init__(self,src,k):
        self.src=src
        self.k=k
        self.centersrc=self.centerlize()
        self.cromartix=self.cormartix1()








    def centerlize(self):
        # 1. 求矩阵中心化后的值
        srcMean=[ np.mean(attr)  for attr in self.src.T ]

        #注意要更新src值

        self.src=self.src-srcMean
        print(f"中心化的矩阵是：{self.src}")

        return self.src

    def cormartix1(self):

        #2. 求特征的协方差矩阵 和 特征矩阵  注：特征指得是图片得属性，例如一个人有鼻子，有嘴都是特征
        # 根据公式 D = ZT * Z *1/(M-1)
        #协方差矩阵
        cormatix=np.dot(self.centersrc.T,self.centersrc) / (self.src.shape[0]-1)

        #特征值和特征矩阵
        value,featuremartix=np.linalg.eig(cormatix)
        #按照从小到大的进行排序，因为取了负号，最大值为
        print(f"特征值是  {value}")
        index = np.argsort(-1*value)
        print(f"按照特值大小排序后的下标是  {index}")


        print(f"特征矩阵是  {featuremartix}")

        # 这里得到的矩阵是  2*3 的 需要转置
        resultmatrx = [featuremartix[:,index[i]] for i in range(self.k)]

        resultmatrx=np.transpose(resultmatrx)


        return  resultmatrx

    def result(self):
        res=np.dot(self.src,self.cromartix)

        return res


class PCAnew:
    def __init__(self,src,k):
        self.src=src
        self.k=k

    def method(self):

        # 将矩阵中心化
        self.src = self.src -  self.src.mean(axis=0)
        # 构建协方差矩阵
        covmatrix = np.dot(self.src.T,self.src)/self.src.shape[0]

        #特征值和特征矩阵
        value,featurematrix = np.linalg.eig(covmatrix)

        idx=np.argsort(-1*value)

        #取前k维得特征向量
        resultmatrix = featurematrix[:,idx[:self.k]]


        return  np.dot(self.src,resultmatrix)







if __name__ == '__main__':

    x =np.array([
    [1,10,2],
    [3, 14, 0],
    [6, 1, 4],
    [21, 0, 8],
    [5, 5, 9],
    [8, 33, 13],
    [17, 7, 6],
    [22, 7, 4],
    [14, 9, 13],
    [18, 12, 7],
    [29, 19, 7]
])


    #原版PCA
    test=PCAorgin(x,2)
    res=test.result()
    print(f"结果矩阵是：{res}")




    #精简版PCA
    # test1 = PCAnew(x,2)
    # result = test1.method()
    # print(result)


    #接口版PCA
    # test2 = pca(n_components=2)
    # test2.fit(x)
    # new_Data = test2.fit_transform(x)
    # print(new_Data)

    #PCA实例

    # x,y=load_iris(return_X_y=True)
    # pca = dp.PCA(n_components=2)
    # reduce_x = pca.fit_transform(x)
    # red_x,red_y=[],[]
    # green_x,green_y=[],[]
    # blue_x,blue_y=[],[]
    #
    #
    # print(f"降维前原始数据形状是：{x.shape}")
    #
    # print(f"reduce_x降维后的数据形状是：{reduce_x.shape}")
    # for i in range(len(reduce_x)):
    #     if y[i]==0:
    #         red_x.append(reduce_x[i][0])
    #         red_y.append(reduce_x[i][1])
    #     elif y[i]==1:
    #         green_x.append(reduce_x[i][0])
    #         green_y.append(reduce_x[i][1])
    #     else:
    #         blue_x.append(reduce_x[i][0])
    #         blue_y.append(reduce_x[i][1])
    # plt.scatter(red_x,red_y,c='r',marker='X')
    # plt.scatter(green_x,green_y,c='g',marker='D')
    # plt.scatter(blue_x,blue_y,c='b',marker='.')
    # plt.show()
















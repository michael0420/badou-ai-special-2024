import numpy as np

class PCA(object):
    """
    PCA简化版实现
    """

    def __init__(self, X, k, is_print=True):
        self.X = X
        self.k = k

    def to_PCA(self):
        # 中心化
        X = self.X - self.X.mean(axis=0)
        print("中心化",X)
        # 协方差
        xfc = np.dot(X.T,X)/(X.shape[0]-1)
        print("协方差",xfc)
        # 计算特征值和特征向量
        tzz,tzxl = np.linalg.eig(xfc)
        print("特征值",tzz)
        print("特征向量",tzxl)
        # 对特征值从大到小排序，结果是索引值按特征值的排序
        index_sort=np.argsort(-tzz)
        # 取前k个特征向量
        tzxl=tzxl[:,index_sort[:self.k]]
        print("取k后特征向量",tzxl)
        # 原数据*特征向量=降维后的数据
        result=np.dot(self.X,tzxl)

        return result
    
def test_01():
    res_01 = np.array(
        [
            [-13.46265879, -0.14716812],
            [21.26163019, -6.12047583],
            [-4.72218421, 11.17511862],
            [-20.73656976, 4.11279645],
            [29.35392285, 16.6403498],
            [24.34524952, -15.35505662],
            [-2.02368689, -6.94159433],
            [-17.20180383, -7.68072922],
            [-12.59724119, -2.8162366],
            [-4.2166579, 7.13299586],
        ]
    )

    res_02 = np.array(
        [
            [-4.56200104, -21.2336912],
            [-39.28629002, -15.26038349],
            [-13.30247561, -32.55597794],
            [2.71190993, -25.49365577],
            [-47.37858268, -38.02120912],
            [-42.36990935, -6.0258027],
            [-16.00097294, -14.43926499],
            [-0.822856, -13.7001301],
            [-5.42741864, -18.56462272],
            [-13.80800193, -28.51385518],
        ]
    )

    res = res_01 - res_02
    print(res)


# test_01()
    
def yuanweihua_test():
    import matplotlib.pyplot as plt
    import sklearn.decomposition as dp
    from sklearn.datasets import load_iris

    x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签

    # 原始数据
    print(x,y.reshape(-1,1))

    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    green_x,green_y=[],[]
    for i in range(len(x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i]==0:
            red_x.append(x[i][0])
            red_y.append(x[i][1])
        elif y[i]==1:
            blue_x.append(x[i][0])
            blue_y.append(x[i][1])
        else:
            green_x.append(x[i][0])
            green_y.append(x[i][1])

    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')
    plt.show()

    pca=PCA(X=x,k=2) #加载pca算法，设置降维后主成分数目为2
    reduced_x=pca.to_PCA()

    print(reduced_x)

    reduced_x-=np.mean(reduced_x,axis=0)

    print(reduced_x)

    # 将第二列值*-1
    reduced_x[:,1]=-1*reduced_x[:,1]

    print("最终结果：\n",reduced_x)

    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    green_x,green_y=[],[]
    for i in range(len(reduced_x)): #按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i]==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])

    # 展示散点图，可视化数据分布情况，其实就是降维后两个特征的可区分度
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')
    plt.show()

yuanweihua_test()

def guiyi_test():
    matrix = np.array(
        [
            [-4.56200104, -21.2336912],
            [-39.28629002, -15.26038349],
            [-13.30247561, -32.55597794],
            [2.71190993, -25.49365577],
            [-47.37858268, -38.02120912],
            [-42.36990935, -6.0258027],
            [-16.00097294, -14.43926499],
            [-0.822856, -13.7001301],
            [-5.42741864, -18.56462272],
            [-13.80800193, -28.51385518],
        ]
    )
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)

    mean_val = np.mean(matrix)
    std_val = np.std(matrix)
    normalized_matrix = (matrix - mean_val) / std_val

    print(normalized_matrix)

# guiyi_test()


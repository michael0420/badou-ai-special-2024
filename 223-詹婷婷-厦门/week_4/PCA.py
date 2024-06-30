import numpy as np
def zero_mean(Z):
    print(Z.T)
    mean = np.array([np.mean(attr) for attr in Z.T]) #求每个维度的均值
    print(mean)
    centrZ = Z - mean #矩阵中心化
    print(centrZ)
    return centrZ

def cov(Z):
    m = Z.shape[0]
    covZ = np.dot(Z.T,Z) / (m-1)  #协方差矩阵 covZ = np.dot(Z.T,Z) / (m-1)
    print(covZ)
    return covZ

def _U(covZ,k):
    #求Z的协方差矩阵covZ的特征值a，和特征向量b
    a,b = np.linalg.eig(covZ) #对协方差矩阵求特征值和特征向量
    print(a)
    print(b)

    ind = np.argsort(-1 * a)  #对特征值按照从大到小的顺序排列
    print(ind)
    UT = [b[:,ind[i]] for i in range(k)] #选择其中最大的k个特征值，将其对应的k个特征向量分别作为列向量组成的特征向量矩阵
    print(UT)
    U = np.transpose(UT)
    print("-----------------")
    print(U)
    return U


def cpca(Z,U):
    print("*****************")
    print(Z)
    pca_z = np.dot(Z,U) #将数据集Z，投影到选取的特征向量上，得到新的已经降维的数据集
    print(pca_z)
    return pca_z


if __name__ == '__main__':
    Z = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    print(Z)
    zero_mean_z = zero_mean(Z) #先对原始数据中心化
    covZ = cov(zero_mean_z)  #求协方差矩阵
    U = _U(covZ, Z.shape[1]-1) #对协方差矩阵求特征向量，这些特征向量组成了新的特征空间
    cpca(zero_mean_z,U)#将数据集Z，投影到选取的特征向量上，得到新的已经降维的数据集

# [[-13.46265879  -0.14716812]
#  [ 21.26163019  -6.12047583]
#  [ -4.72218421  11.17511862]
#  [-20.73656976   4.11279645]
#  [ 29.35392285  16.6403498 ]
#  [ 24.34524952 -15.35505662]
#  [ -2.02368689  -6.94159433]
#  [-17.20180383  -7.68072922]
#  [-12.59724119  -2.8162366 ]
#  [ -4.2166579    7.13299586]]
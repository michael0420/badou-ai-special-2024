import  numpy as np

class PCA():
    def __init__(self,np_size):
        self.np_size = np_size
    def transform_features(self,X):

        X = X - X.mean(axis=0)
        # 求协方差矩阵
        self.covariance = np.dot(X.T,X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:,idx[:self.np_size]]
        # 对X进行降维
        return np.dot(X,self.components_)

if __name__ == '__main__':
    pca = PCA(np_size=2)
    X = np.array([[6, 25, 18],
                  [16, 33, 55],
                  [26, 43, 24],
                  [15, 9, 46],
                  [52, 41, 21],
                  [9, 48, 5],
                  [8, 9, 13],
                  [81, 56, 55],
                  [13, 31, 45],
                  [40, 41, 23]])
    newX=pca.transform_features(X)
    print("降维后矩阵为：")
    print(newX)                  #输出降维后的数据
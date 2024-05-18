'''
【第四周作业】

作业：1.实现高斯噪声 2.实现椒盐噪声 3.实现PCA  4.拓展：证明中心化协方差矩阵公式

'''

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets._base import load_iris
import numpy as np

# 3.实现PCA
# 使用PCA求样本矩阵X的K阶降维矩阵Z
# 一般步骤是这样的：
# 1. 对原始数据零均值化（中心化），
# 2. 求协方差矩阵，
# 3. 对协方差矩阵求特征向量和特征值，
# 4.用得出的特征向量值进行降序排序（从大到小）
# 5。取出前K个特征向量分别作为列向量组成特征向量矩阵W。
#6将原始数据data*W得到降维后的k维新矩阵Z
#
# 准备原始数据
# np_list=np.random.randint(low=1,high=30,size=(10,3))
# print(np_list)
data = np.array([[13, 17, 2],
                 [25, 10, 4],
                 [15, 14, 26],
                 [24, 17, 16],
                 [20, 24, 3],
                 [4, 5, 3],
                 [11, 11, 2],
                 [5, 3, 16],
                 [16, 7, 14],
                 [26, 19, 7]])
# 定义K的降维数
K = np.shape(data)[1] - 1
# 1. 对原始数据零均值化（中心化），
zxhdata = []
mean=[]
'''
for attr in self.X取的是X里的每一行，
for attr in self.X.T取的就是X.T转置矩阵的每一行，
就变相取的X的每一列，attr是循环变量也就是每一列的数据，
np.mean(attr)计算结果就是列的平均值
对原始数据零均值化
理解：求样本集中的每个维度上的特征均值
公式 样本集的每个样本第一维相加求平均sum[样本1[0]+样本2[0]+...样本n[0]]/sum
    样本集的每个样本第二维相加求平均sum[样本1[1]+样本2[1]+...样本n[1]]/sum
    样本集的每个样本第三维相加求平均sum[样本1[2]+样本2[2]+...样本n[2]]/sum
然后每个样本中的每个数减去所属维度的特征均值得到的新矩阵就是原数据中心化的后的矩阵
'''
for attr in data.T:
    # print(attr)
    mean1=np.mean(attr)
    mean.append(mean1)
# print(mean)
# 原数据中心化的后的矩阵
zxhdata=data-mean
# print(zxhdata)
# 2. 求协方差矩阵，
#样本集的样例总数
datanum = np.shape(zxhdata)[0]
#样本矩阵的协方差矩阵covariance
covariance= np.dot(zxhdata.T, zxhdata)/(datanum - 1)
#先求X的协方差矩阵covariance的特征值和特征向量
# print(covariance)
#特征值赋值给character
# ，对应特征向量 赋值给feature。
# 函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
character,feature = np.linalg.eig(covariance)# 返回对于中心化矩阵的特征值和特征向量
# 4.用得出的特征向量值进行降序排序（从大到小）
ind = np.argsort(-1*character)
#5.取出前K个特征向量分别作为列向量组成特征向量矩阵W。
# UT = [b[:,ind[i]] for i in range(self.K)]
W=[]
for i in range(K):
    # print(feature[:, ind[i]])
    W.append(feature[:, ind[i]])
#W是2*3的矩阵不能与原矩阵10*3结合需要转置成3*2的矩阵W1
W1 = np.transpose(W)
# print(W1)
# 6将原始数据data*W1得到降维后的k维新矩阵Z
Z=np.dot(data,W1)
print(Z)
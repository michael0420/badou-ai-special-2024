# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 15:35
# @Author: iris
import numpy as np

if __name__ == '__main__':
    '30样本5特征的样本集, 行为样例，列为特征维度'
    # 设置种子，可以重复实验
    np.random.seed(0)
    # 初始化数据
    X = np.random.uniform(1, 10, (50, 5))
    K = np.shape(X)[1] - 2
    """步骤①. 均值化"""
    mean = np.mean(X, axis=0)
    centerX = np.subtract(X, mean)
    """步骤②. 求协方差矩阵"""
    # 样本集的样例总数
    m = np.shape(centerX)[0]
    # 带入公式求样本矩阵的协方差矩阵D
    # dot 矩阵乘
    D = 1 / m * np.dot(centerX.T, centerX)
    """步骤③. 通过协方差矩阵求特征值(vals)、特征向量(vecs)"""
    vals, vecs = np.linalg.eig(D)
    # 进行降序排序
    index = np.argsort(-vals)
    # 取出前K个值
    UT = [vecs[:, index[i]] for i in range(K)]
    U = np.transpose(UT)
    """步骤④. 获取降维矩阵U"""
    Z = np.dot(X, U)
    print('样本矩阵X的降维矩阵Z:\n', Z)

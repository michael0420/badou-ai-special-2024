import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from P1anpian_PCA import diy_PCA
from sklearn.datasets import load_iris

# 使用iris数据集 展示PCA的结果

# 可视化展示数据集的数据
def show_result(x, name):
    # 红蓝绿不同颜色表示不同的分类类别
    # 横纵坐标分别为降维后的两个特征
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(x)):  # 按鸢尾花的类别将降维后的数据点保存在不同的表中
        if y[i] == 0:
            red_x.append(x[i][0])
            red_y.append(x[i][1])
        elif y[i] == 1:
            blue_x.append(x[i][0])
            blue_y.append(x[i][1])
        else:
            green_x.append(x[i][0])
            green_y.append(x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.title(name)
    plt.show()

if __name__ == '__main__':
    # 加载数据 x为属性 y为标签
    x, y = load_iris(return_X_y= True)

    '''使用sklearn库的PCA'''
    # 加载PCA算法 降维后主成分数目为2
    pca_sklearn = dp.PCA(n_components=2)
    # 对原始数据进行降维 结果保存在reduced_x中
    reduced_x_sklearn = pca_sklearn.fit_transform(x)

    '''使用DIY_PCA'''
    diy_pca = diy_PCA(x, 2)
    reduced_x_diyPCA = diy_pca.Z # 将结果传出

    '''原始数据分析'''
    show_result(reduced_x_sklearn, "pca_sklearn")
    show_result(reduced_x_diyPCA, "diy_pca")

    # 注：在diyPCA中使用的是原始的样本矩阵X和特征向量矩阵U相乘
    # 在sklearn中则是直接使用的中心化后的矩阵去降维 所以sklearn的结果也是中心化的矩阵
"""
@author: 207-xujinlan
ransac算法实现
"""

import numpy as np
import scipy.linalg as sl


def make_data(n_samples, n_inputs=1, n_outputs=1, n_outlines=100):
    """
    创建数据
    :param n_samples: 样本个数，行数
    :param n_inputs: 输入数据列数
    :param n_outputs: 输出数据列数
    :param n_outlines: 局外点个数
    :return: 返回输入数据和输出数据
    """
    x_data = 30 * np.random.random(size=(n_samples, n_inputs))
    k = 60 * np.random.random(size=(n_inputs, n_outputs))
    y_data = np.dot(x_data, k)
    # 添加高斯噪声
    x_data = x_data + np.random.normal(size=x_data.shape)
    y_data = y_data + np.random.normal(size=y_data.shape)
    # 添加局外点
    all_indx = np.arange(x_data.shape[0])
    np.random.shuffle(all_indx)
    outlines_indx = all_indx[:n_outlines]
    x_data[outlines_indx] = 20 * np.random.random(size=(n_outlines, n_inputs))
    y_data[outlines_indx] = 50 * np.random.normal(size=(n_outlines, n_outputs))
    return x_data, y_data


class LinearLeastSquareModel:
    """
    最小二乘法模型
    """
    def __init__(self):
        self.m = None

    def fit(self, x_data, y_data):
        """
        模型训练
        :param x_data: 自变量
        :param y_data: 应变量
        :return:
        """
        self.m, resids, rank, s = sl.lstsq(x_data, y_data)

    def predict(self, x_data):
        """
        模型预测
        :param x_data: 自变量
        :return:
        """
        predict = np.dot(x_data, self.m)
        return predict


def ransac(x_data, y_data, model, n, k, t, d):
    """
    :param x_data: 自变量
    :param y_data: 因变量
    :param n:生成模型所需的最少样本点数
    :param k:最大迭代次数
    :param t:阈值
    :param d:拟合较好时所需的最少样本点数
    :return:best_model 返回内群数最多拟合对应的模型
    """
    i = 0  # 迭代次数初始化
    indxs = np.arange(x_data.shape[0])
    best_model = []   # 存放模型
    best_points = []   #存放内群数

    while i < k:
        # 打乱索引
        np.random.shuffle(indxs)
        # 选生成模型的点
        fit_points_x = x_data[indxs[:n]]
        fit_points_y = y_data[indxs[:n]]
        # 选出剩余的测试点
        test_points_x = x_data[indxs[n:]]
        test_points_y = y_data[indxs[n:]]
        # 模型拟合
        model.fit(fit_points_x, fit_points_y)
        if model.m not in best_model:   #如果模型不在之前的模型中就继续计算，如果在之前的模型中就进入下一次循环
            # 模型预测
            result = model.predict(test_points_x)
            # 计算误差
            erros_per_point = np.sum((result - test_points_y) ** 2, axis=1)
            #
            inliners = test_points_y[erros_per_point < t].shape[0]  # 筛选误差小于阈值的点的个数
            if inliners > d:  # 如果内群数大于阈值，就记下内群数量值和对应的模型
                best_points.append(d)  # 将群内点数记下
                best_model.append(model.m)  # 记下相应模型
        i += 1
    return best_model[np.argmax(best_points)]  # 返回内群点数最多对应的模型


if __name__ == '__main__':
    x_data, y_data = make_data(n_samples=500, n_inputs=1, n_outputs=1, n_outlines=100)
    llsm = LinearLeastSquareModel()  # 实例化模型
    m = ransac(x_data, y_data, llsm, n=5, k=100, t=10, d=20)

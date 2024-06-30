# coding = utf-8

'''
        实现RANSAC|最小二乘法
'''

import numpy as np
import scipy as sp
import scipy.linalg as sl


def start():
    n_data = 500
    n_inputs = 1
    n_outputs = 1
    data = 30 * np.random.random((n_data, n_inputs))     # 生成样本数据
    # print(data.shape, data)
    slope = 30 * np.random.normal(size=(n_inputs, n_outputs))      # 生成斜率
    # print(slope.shape, slope)
    ls = sp.dot(data, slope)        # y=kx
    # print(ls)

    # 高斯噪声处理
    x_noisy = data + np.random.normal(size=data.shape)      # Xi
    # print(x_noisy.shape)
    y_noisy = ls + np.random.normal(size=ls.shape)          # Yi
    # print(y_noisy.shape)

    # 离群数据处理
    if 1:
        n_outliers = 150
        all_idxs = np.arange(x_noisy.shape[0])
        out_idxs = np.random.choice(all_idxs, size=n_outliers, replace=False)       # 随机选择不重复的索引
        # print(out_idxs.shape)
        x_noisy[out_idxs] = 5 * np.random.random((n_outliers, n_inputs))
        y_noisy[out_idxs] = 7 * np.random.normal(size=(n_outliers, n_outputs))
        np.random.shuffle(all_idxs)     # np.random.shuffle()将原数组重新排序

    new_data = np.hstack((x_noisy, y_noisy))      # np.hstack()行相同，列叠加，[x,y]...
    # print(new_data.shape, new_data)

    input_columns = range(n_inputs)     # 创建range对象，生成一个从0开始到n_inputs-1的整数序列
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False

    # scipy.linalg.lstsq 最小二乘法求解
    linear_fit, resids, _, _ = sp.linalg.lstsq(new_data[:, input_columns], new_data[:, output_columns])
    # print(linear_fit)
    # 使用最小二乘法实例代入RANSAC求解
    model = Least_Squares_Method(input_columns, output_columns, debug=debug)
    ransac_fit, ransac_data = ransac(new_data, model, 50, 1000, 5e3, 300, debug=debug)


    if 1:
        import pylab

        sort_idxs = np.argsort(data[:, 0])
        A_col0_sorted = data[sort_idxs]

        # 绘制散点图
        if 1:
            pylab.plot(x_noisy[:, 0], y_noisy[:, 0], 'k.', label='data')
            pylab.plot(x_noisy[ransac_data['inliers'], 0], y_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")

            pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')      # 绘制RANSAC拟合结果
            pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, slope)[:, 0],
                   label='y= kx + b')       # 绘制标准结果
            pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')      # 绘制最小二乘结果
            pylab.legend()
            pylab.show()

    return new_data, model


class Least_Squares_Method():

    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack()使列相同，行叠加
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        x, resids, _, _ = sl.lstsq(A, B)  # scipy.linalg.lstsq 求解线性最小二乘问题，residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-->行Yi
        B_fit = sp.dot(A, model)  # 计算y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point  # 返回误差


def ransac(data, model, n, k, t, d, debug=False):
    # data 样本数据
    # model 模型
    # n 生成模型所需最少样本点
    # k 最大迭代数
    # t 自定义阈值，判断满足模型的条件
    # d 拟合较好时，需要的最少样本点数
    iterations = 0  # 迭代计数
    bestfit = None  # 最佳匹配
    besterr = np.inf  # 误差设置默认值，np.inf表示正无穷大的特殊浮点数值
    best_inlier_idxs = None     # 内群点
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])      # 获取随机可能内群点
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差
        other_idxs = test_idxs[test_err < t]     # 判断误差是否小于阈值
        other_inliers = data[other_idxs, :]

        print(f'最少需满足 {d} 个内群点，当前内群数：{len(other_inliers)}')
        if (len(other_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, other_inliers))  # 样本连接，垂直堆叠数组，确定当前所有内群点
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            # 更新最优解和最小误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, other_idxs))  # 更新内群点
        iterations += 1
    if bestfit is None:
        raise ValueError("暂未匹配到符合条件的模型")
    else:
        return bestfit, {'inliers':best_inlier_idxs}


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


if __name__ == '__main__':
    start()


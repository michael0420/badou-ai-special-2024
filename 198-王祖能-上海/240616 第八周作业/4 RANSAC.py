'''
@zeno wang
随机采样一致性random samples consensus，基于模型一致，优化出类穷举找到能包含最多样本点的模型函数
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as sl
import random


# 设计样本点
def find(n):
    # 1.生成理想数据
    samples_num = n  # 样本数
    inputs_num = 1  # 输入变量个数
    outputs_num = 1  # 输出变量个数
    X_exact = 20 * np.random.random([samples_num, inputs_num])  # 随机生成n个0-20以内数
    K_exact = 60 * np.random.normal(loc=0, scale=1, size=[inputs_num, outputs_num])  # 符合正态分布的随机斜率，均值loc方差scale
    print('理想初始化斜率为：', K_exact)  # 随机初始化一个斜率
    Y_exact = np.dot(X_exact, K_exact)  # y = k * x  而且为什么sp.dot无法找到?
    print('理想数据为：\n', np.hstack([X_exact, Y_exact]))  # x,y并列表示

    # 2.随机加入高斯噪声，最小二乘可以处理掉
    # X_noise, Y_noise = X_exact + random.gauss(0, 1), Y_exact + random.gauss(0, 1)  # 错误方法，所有样本增减相同的高斯随机数，整体偏移
    X_noise, Y_noise = X_exact + np.random.random(size=[samples_num, inputs_num]), Y_exact + np.random.random(size=Y_exact.shape)
    # 老师方法    A_noisy = A_exact + np.random.normal( size = A_exact.shape ) #500 * 1向量,代表Xi
    print('添加噪声点数据为：\n', np.hstack([X_noise, Y_noise]))

    # 3.添加外群点
    if 1:  # debug方便寻找程序报错位置
        outpoints_num = int(0.2 * n)  # 外群点数
        all_index = np.arange(samples_num)  # 生成样本数的序列号，np.arange(3)返回结果是【0， 1， 2】的序列
        np.random.shuffle(all_index)  # 序列号随机打乱
        outpoints_index = all_index[:outpoints_num]
        print('所有点索引为：{}，外群点索引为：{}'.format(all_index, outpoints_index))
        X_noise[outpoints_index] = 20 * np.random.random([outpoints_num, inputs_num])
        Y_noise[outpoints_index] = 50 * np.random.random(size=[outpoints_num, inputs_num])

    # 4.形成最终数据
    data = np.hstack([X_noise, Y_noise])  # 500行2列的全部样本数据
    input_data = data[:, :inputs_num]
    output_data = data[:samples_num, inputs_num:]
    print('添加外群点、噪声点的数据为：\n', data)
    print('自变量数据为：\n{}\n因变量数据为：\n{}'.format(input_data, output_data))
    # plt.scatter(X_noise, Y_noise)
    # plt.show()

    # 5.建立模型
    debug1, debug2, return_all, plot = 0, 1, True, True
    # 方法一：不做任何数据处理，直接最小二乘法求线性
    K_lst, residues, rank, s = sl.lstsq(input_data, output_data)
    # 方法二：Ransac结合最小二乘求线性
    model = LinearLeastSquareModel(input_data, output_data, debug1)  # 调用类，最小二乘类的实例化生成模型
    x = model.slope(data)
    if K_lst == x:
        print('LST类调用无误，斜率为：', K_lst)
    else:
        print('LST类调用错误')
    K_ransac, err_ransac, idx_fit = ransac(data, model, n, iter=1000, threshold=1.5e4, debug=debug1, return_all=return_all)
    print('Ransac拟合相关关系为：{},均方差为：{}\n内群点索引为：\n{}'.format(K_ransac, err_ransac, idx_fit))


    # 7.对比绘图
    if plot:
        import pylab as pyl
        sort_idx = np.argsort(X_exact[:, 0])  # 从小到大排列获取对应的索引编号，样本作为X坐标升序排列，(lexsort)
        X_sort = X_noise[sort_idx]
        # pyl.rcParams['font.sans-serif'] = ['SimHei']

        pyl.scatter(X_noise, Y_noise, s=20, c='black', marker='^', label='data with noise')  # 1.散点图，绘制带噪声真实样本分布
        # pyl.plot(X_noise, Y_noise, 'k.', label='data')  # 仍为绘制散点图, k表示黑色，.表示原点
        pyl.plot(X_noise[idx_fit['samples_fit_idx'], 0], Y_noise[idx_fit['samples_fit_idx'], 0], 'b+', label='data_ransac')  # 2.散点图，绘制ransac结果处理的最佳点
        pyl.plot(X_sort[:, 0], np.dot(X_sort, K_exact)[:, 0], label='exact line')
        pyl.plot(X_sort[:, 0], np.dot(X_sort, K_lst)[:, 0], label='lst line')
        pyl.plot(X_sort[:, 0], np.dot(X_sort, K_ransac)[:, 0], label='ransac line')
        pyl.title('Ransac'), pyl.legend()
        pyl.show()

        if debug2:  # 检查增加噪音对数据的影响
            pyl.plot(X_exact, Y_exact, 'r.', label='data_exact')
            pyl.plot(X_noise, Y_noise, 'g+', label='data_noise')
            pyl.legend()
            pyl.show()


    # 6.RANSAC思想
def ransac(data, model, n, iter, threshold, debug, return_all):
    '''
    :param data:  样本数据
    :param model:  模型实例化
    :param n:  样本总数
    :param iter:  最大迭代次数
    :param threshold:  采用最小二乘法计算所有样本累计误差的阈值，比此值小认为足够收敛，拟合较好
    :param debug:
    :param return_all:
    :return:  最优的线性拟合值
    '''
    n_init = int(0.1 * n)      # param n_init:  第一次随机初始化选取的样本点个数
    n_lst = int(0.6 * n)       # param n_lst: 不仅满足拟合条件，还至少要包含样本的数量
    iterations = 0
    k_best = None
    err_best = np.inf  # 初始误差设为无限大值，不断优化向下收敛
    idx_best = []
    while iterations < iter:
        idx_init, idx_test = idx_partition(n_init, data.shape[0])  # 设定随机初始拟合模型的样本序号，以及待测试序号
        print('随机初始样本序号为：\n{}\n待测试样本序号：\n{}'.format(idx_init, idx_test))
        data_init = data[idx_init, :]  # 随机初始样本数据[Xi, Yi]
        data_test = data[idx_test, :]
        k_may = model.slope(data_init)  # 随机初始数据最小二乘拟合斜率
        err_test = model.get_error(data_test, k_may)  # 剩余测试点的累计误差
        print('error_test=', err_test < threshold)  # 误差小于阈值时输出True
        idx_also = idx_test[err_test < threshold]
        print('idx_may=', idx_also)
        data_also = data[idx_also, :]
        if debug:
            print('data_test min error:', min(err_test))
            print('data_test max error:', max(err_test))
            print('data_test average error:', np.mean(err_test))
        if (len(data_init) + len(data_also)) > n_lst:
            data_better = np.concatenate([data_init, data_also], axis=0)  # 按行拼接矩阵
            k_better = model.slope(data_better)
            err_better = np.mean(model.get_error(data_better, k_better))
            if err_better < err_best:
                k_best = k_better
                err_best = err_better
                idx_best = np.concatenate([idx_init, idx_also], axis=0)
        print('当前迭代次数为：%d' % (iterations+1))
        iterations += 1
    if k_best == None:
        return ValueError('DIDNT FIND')
    if return_all:
        return k_best, err_best, {'samples_fit_idx': idx_best}
    else:
        return k_best


def idx_partition(part, sample_num):  # 原数据切分为两部分，初始随机选取进行拟合得到猜测模型，然后判断剩余点是否在阈值误差内
    all_index = np.arange(sample_num)  # 获取下标索引，并随机打乱分割两部分
    np.random.shuffle(all_index)
    idx1 = all_index[:part]
    idx2 = all_index[part:]
    return idx1, idx2


class LinearLeastSquareModel:
    def __init__(self, in_data, out_data, debug):
        self.in_data = in_data
        self.out_data = out_data
        self.debug = debug
        self.in_col = self.in_data.shape[1]  # 输入变量列数，即个数
        self.out_col = self.out_data.shape[1]  # 输出变量列数，即个数

    def slope(self, data):
        temp1 = np.vstack([data[:, i] for i in range(0, self.in_col)]).T  # 这个就是data[:,:sel.in_col]切片，只是一维不能带入sl.lstsq
        temp2 = np.vstack([data[:, i] for i in range(self.in_col, self.in_col + self.out_col)]).T  # data[：, i]是个一维向量，形状为(n,)，水平显示，堆叠为上下叠加才形成矩阵形式
        x, residues, rank, s = sl.lstsq(temp1, temp2)  # x斜率，residues二范数残差和, y=kx+b入参必须是行向量
        return x  # 返回最小二乘法得到的斜率

    def get_error(self, data, k):
        temp1 = np.vstack([data[:, i] for i in range(0, self.in_col)]).T
        temp2 = np.vstack([data[:, i] for i in range(self.in_col, self.in_col + self.out_col)]).T
        data_fit = np.dot(temp1, k)  # 线性问题k应该就是所求的斜率
        error_per_row = np.sum((temp2 - data_fit) ** 2, axis=1)  # 0是第一个维度row求和，对行中各列元素求和，1是第二个维度对竖向求和。temp已经是行向量了，[i, j]是各点的误差向量
        return error_per_row


if __name__ == '__main__':
    find(500)

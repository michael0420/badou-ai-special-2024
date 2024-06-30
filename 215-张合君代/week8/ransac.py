# -*- coding: utf-8 -*-
"""
@author: zhjd
Least Square Method and RANSAC procedure and code optimization
"""
import numpy as np
import pylab
import scipy.linalg as sl


def least_square(array_2d):
    """
    Least Square Method
    :param array_2d: 2d array contains all x-y values ((x1,y1),(x2,y2),(x3,y3)...)
    :return: k,b
    """
    xi, yi = array_2d[:, 0], array_2d[:, 1]
    sum_xi, sum_yi = np.sum(xi), np.sum(yi)
    sum_xi_sq = np.sum(xi ** 2)
    sum_xiyi = np.sum(xi * yi)
    size = len(array_2d)

    denominator = size * sum_xi_sq - sum_xi ** 2
    k = (size * sum_xiyi - sum_xi * sum_yi) / denominator
    b = (sum_yi * sum_xi_sq - sum_xi * sum_xiyi) / denominator
    return k, b


def fit(model, A, b):
    if model is None:
        raise ValueError("model is not initiated")

    M = A[:, np.newaxis] ** (0, 1) if model.include_intercept else A[:, np.newaxis]
    x, residues, _, _ = sl.lstsq(M, b)  # python语法：对不使用的参数用占位符，提高代码可读性和可维护性，不影响执行效率
    # scipy语法：没有截距的模型，x长度为1，所以以下代码需要加判断
    model.slope = x[1] if model.include_intercept else x[0]
    model.intercept = x[0] if model.include_intercept else 0
    return x, residues


def check_residual(model, threshold, A, b, indexes):
    residuals = (A[indexes] * model.slope + model.intercept - b[indexes]) ** 2
    return indexes[residuals < threshold]


class Ransac:
    def __init__(self, A, b, model, iterations, n, threshold, min_inliers):
        """
        RANSAC algorithm
        :param A: Independent variable array, 1D array
        :param b: Dependent variable array, 1D array
        :param model: Model to fit
        :param iterations: Number of iterations
        :param n: Number of samples to fit per iteration
        :param threshold: Residual threshold to determine inliers
        :param min_inliers: Minimum number of inliers to accept a model
        """
        if len(A) != len(b):
            raise ValueError("Array A and b must have the same length")
        self.A, self.b = A, b
        self.model = model
        self.iterations = iterations
        self.n = n
        self.threshold = threshold
        self.min_inliers = min_inliers

    def run(self):
        best_residue, best_fit, best_fit_inliers = np.inf, None, None

        # python语法：下划线占位符代替未使用的参数
        for _ in range(self.iterations):
            # 1. 每次从数据集中去n组
            sample_indexes = np.random.choice(len(self.A), self.n, replace=False)
            # 2. 用最小二乘法拟合，获得k,b
            fit(self.model, self.A[sample_indexes], self.b[sample_indexes])
            # 3. 根据拟合结果计算其他数据残差，小于阈值视为内群数据，否则为外群数据
            inliers = check_residual(self.model, self.threshold, self.A, self.b, np.arange(len(self.A)))
            # 3.1. 根据判定的内群数据，数量不足则跳过本次循环
            if len(inliers) < self.min_inliers + len(sample_indexes):
                continue

            # 3.2. 根据判定的内群数据再次拟合，获得模型和残差
            x, residues = fit(self.model, self.A[inliers], self.b[inliers])
            if residues < best_residue:
                # 4. 比较iterations次迭代后最小残差
                best_residue, best_fit, best_fit_inliers = residues, x, inliers

        if best_fit is None:
            raise ValueError("did't meet fit acceptance criteria")

        return best_fit, best_fit_inliers


class LinearModel:
    def __init__(self, include_intercept=True):
        """
        Linear function model
        :param include_intercept: Whether to include intercept, True for y=kx+b, False for y=kx
        """
        self.include_intercept = include_intercept
        self.slope = None
        self.intercept = None


if __name__ == '__main__':
    # least square 公式
    # print(least_square(((1, 6), (2, 5), (3, 7), (4, 10))))

    # RANSAC
    data_size = 500
    data_range = 50
    outer_size = int(data_size * 0.2)
    # 1. 随机生成一个线性函数斜率和截距作为的目标，y=kx+b
    k = np.random.uniform(0, data_range)
    b = np.random.uniform(0, data_range)
    true_params = [0, k]
    # true_params = [b, k]

    # 2. 模拟出观测值作为内群数据，在正确的线性函数自变量和因变量上加高斯噪声
    x_array = np.random.uniform(0, data_range, size=data_size)
    # y_array = np.dot(x_array, true_params) 行数与列数不一致，无法运用点乘
    y_array = x_array * k + b

    # 2.2. 加入高斯噪声后
    x_array_noise = x_array + np.random.normal(size=data_size)
    y_array_noise = y_array + np.random.normal(size=data_size)

    # 3. 模拟离群数据
    # 3.1. 在自变量数组中随机索引位置上替换为随机数
    outliers = np.random.choice(data_size, outer_size, replace=False)
    x_array_noise[outliers] = np.random.uniform(0, data_range, size=outer_size)
    # 噪声类型：均匀噪声与高斯噪声
    # 像高斯噪声这样，数值集中在一个中心值周围，使得离群数据在所有点中分布集中，而不是像均匀噪声那样，离群数据分布分散、均匀。
    # 由于RANSAC随机采样的策略，当离群数据集中分布，每次都抽取到噪声的概率相较于均匀分布的低一些，所以更能准确识别外群数据。
    # y_array_noise[outliers] = np.random.uniform(0, -50, size=outer_size)
    y_array_noise[outliers] = 50 * np.random.normal(size=outer_size)

    # 4. 使用 RANSAC思想和scipy lstsq算法拟合，比对拟合结果和目标
    # 不含截距的拟合效果更好
    model = LinearModel(include_intercept=False)
    # 科学计数法 7e3 表示 7 × 10^3
    ransac = Ransac(x_array_noise, y_array_noise, model, 1000, 50, 5e3, 300)
    best_fit, best_fit_inliers = ransac.run()

    # 加一列1的实现方式：
    # np.vstack([x_array_noise, np.ones(data_size)]).T 基于数组操作：stack和转置2步操作，存在内存分配。
    # x_array[:, np.newaxis] ** [0, 1] 基于传播和元素级操作，理论上说数据规模越大性能优势越明显。
    # 广播Broadcasting，Numpy重要思想，让不同shape的数组计算更灵活方便，
    # 就像让不同数组的值broadcast出去，而忽略数组的shape，关注值的运算本身。

    # 对比区别：
    # np.vstack([self.A, self.b]).T 数据量大的情况下转置成本较高，不如直接进行准确的行/列操作
    # np.column_stack((self.A, self.b))
    if model.include_intercept:
        linear_fit, _, _, _ = sl.lstsq(x_array_noise[:, np.newaxis] ** [0, 1], y_array_noise)
    else:
        A = np.vstack([x_array_noise]).T
        B = np.vstack([y_array_noise]).T
        linear_fit, _, _, _ = sl.lstsq(A, B)

    sorted_indices = np.argsort(x_array)
    x_sorted = x_array[sorted_indices]
    x_sorted = x_sorted[:, np.newaxis] ** [0, 1]

    pylab.plot(x_array_noise, y_array_noise, 'k.', label='data')
    pylab.plot(x_array_noise[best_fit_inliers], y_array_noise[best_fit_inliers], 'bx', label="RANSAC data")

    if model.include_intercept:
        pylab.plot(x_sorted[:, 1], np.dot(x_sorted, best_fit), label='RANSAC fit')
        pylab.plot(x_sorted[:, 1], np.dot(x_sorted, true_params), label='exact system')
        pylab.plot(x_sorted[:, 1], np.dot(x_sorted, linear_fit), label='linear fit')
    else:
        pylab.plot(x_sorted[:, 1], x_sorted[:, 1] * best_fit[0], label='RANSAC fit')
        pylab.plot(x_sorted[:, 1], x_sorted[:, 1] * true_params[1], label='exact system')
        pylab.plot(x_sorted[:, 1], x_sorted[:, 1] * linear_fit[0], label='linear fit')

    pylab.legend()
    pylab.show()

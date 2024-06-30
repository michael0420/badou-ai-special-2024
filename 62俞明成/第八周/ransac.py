import numpy as np
import matplotlib.pyplot as plot
import matplotlib

matplotlib.use("TkAgg")


def GenTestData():
    n_samples = 500  # 采样点数
    n_input = 1
    n_output = 1
    x_exact = 20 * np.random.random((n_samples, n_input))
    perfect_fit = 60 * np.random.normal(size=(n_input, n_output))  # 随机线性度，即随机生成一个斜率
    y_exact = np.dot(x_exact, perfect_fit)  # y = x * k

    x_noisy = x_exact + np.random.normal(size=x_exact.shape)
    y_noisy = y_exact + np.random.normal(size=y_exact.shape)
    # 添加“局外点”
    n_outliers = 100
    all_index = np.arange(x_noisy.shape[0])
    np.random.shuffle(all_index)
    outlier_index = all_index[:n_outliers]
    x_noisy[outlier_index] = 20 * np.random.random((n_outliers, n_input))
    y_noisy[outlier_index] = 50 * np.random.normal(size=(n_outliers, n_output))

    noisy_data, exact_data = np.concatenate((x_noisy, y_noisy), axis=1), np.concatenate((x_exact, y_exact), axis=1)

    input_columns = range(n_input)  # 数组的第一列x:0
    output_columns = [n_input + i for i in range(n_output)]  # 数组最后一列y:1
    model = LinearLeastSquareModel(input_columns, output_columns)  # 类的实例化:用最小二乘生成已知模型

    linear_fit, resids, rank, s = np.linalg.lstsq(noisy_data[:, input_columns], noisy_data[:, output_columns], rcond=None)

    # run RANSAC 算法
    ransac_fit, ransac_data = ransac(noisy_data, model, 50, 1000, 7e3, 300, return_all=True)

    sort_index = np.argsort(x_exact[:, 0])
    A_col0_sorted = x_exact[sort_index]
    plot.plot(x_noisy[:, 0], y_noisy[:, 0], 'k.', label='data')  # 散点图
    plot.plot(x_noisy[ransac_data['inliners'], 0], y_noisy[ransac_data['inliners'], 0], 'bx', label="RANSAC data")
    plot.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
    plot.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
    plot.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
    plot.legend()
    plot.show()


class LinearLeastSquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = np.linalg.lstsq(A, B, rcond=None)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def ransac(data, model, n, k, t, d, return_all=False):
    iterations = 0
    bestfit = None
    best_err = np.inf
    best_inliner_index = None
    while iterations < k:
        maybe_index, test_index = random_partition(n, data.shape[0])
        maybe_inliners = data[maybe_index, :]
        test_points = data[test_index]
        maybe_model = model.fit(maybe_inliners)
        test_err = model.get_error(test_points, maybe_model)  # 计算误差:平方和最小
        also_index = test_index[test_err < t]
        also_inliners = data[also_index, :]

        if (len(also_inliners) > d):
            betterdata = np.concatenate((maybe_inliners, also_inliners))  # 样本连接
            better_model = model.fit(betterdata)
            better_errs = model.get_error(betterdata, better_model)
            this_err = np.mean(better_errs)  # 平均误差作为新的误差
            if this_err < best_err:
                bestfit = better_model
                best_err = this_err
                best_inliner_index = np.concatenate((maybe_index, also_index))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliners': best_inliner_index}
    else:
        return bestfit


def random_partition(n, n_data):
    """返回n个随机的数据行和其他len(data)-n行"""
    all_index = np.arange(n_data)  # 获取n_data下标索引
    np.random.shuffle(all_index)  # 打乱下标索引
    index1 = all_index[:n]
    index2 = all_index[n:]
    return index1, index2


if __name__ == '__main__':
    GenTestData()

import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab


# 最小二乘法求线性解，用户ransac输入模型
class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack([data[:, col] for col in self.input_columns]).T
        B = np.vstack([data[:, col] for col in self.output_columns]).T
        # 残差和
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        # 计算误差，平方和最小
        A = np.vstack([data[:, col] for col in self.input_columns]).T
        B = np.vstack([data[:, col] for col in self.output_columns]).T
        B_fit = np.dot(A, model)
        return np.sum((B - B_fit) ** 2, axis=1)


# 构建数据
def build_data():
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20之间的500个数据
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        # 添加局外点
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        # 打乱all_idxs
        np.random.shuffle(all_idxs)
        # 100个0-400的随机局外点
        outlier_idxs = all_idxs[:n_outliers]
        # x加入噪声和局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        # y加入噪声和局外点
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

    # 将数据进行合并
    all_data = np.hstack((A_noisy, B_noisy))
    # 获取数组第一列
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]

    # 用最小二乘法生成已知模型
    model = LinearLeastSquareModel(input_columns, output_columns)
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    return A_exact, A_noisy, B_noisy, all_data, model, linear_fit, perfect_fit


def random_partition(n, n_data):
    # 随机选取n个点
    maybe_idxs = np.arange(n_data)
    np.random.shuffle(maybe_idxs)
    maybe_idxs1 = maybe_idxs[:n]
    # 随机选取n个点
    maybe_idxs2 = maybe_idxs[n:]
    return maybe_idxs1, maybe_idxs2


# 实现ransac
def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        # 随机选取n个点
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        # 获取maybe_idxs行数据（xi,yi）
        maybe_inliers = data[maybe_idxs, :]
        # 若干行（Xi,Yi）数据点
        test_points = data[test_idxs]
        # 拟合模型
        maybe_model = model.fit(maybe_inliers)
        # 计算误差：平方和最小
        test_errors = model.get_error(test_points, maybe_model)
        # 筛选误差率小于t的测试集
        also_idxs = test_idxs[test_errors < t]
        # 获取also_idxs行数据（xi,yi）
        also_inliers = data[also_idxs, :]

        if (len(also_inliers) > d):
            # 连接样本
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            # 平均误差作为新的误差
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("没有达到合适的验收标准")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def draw_data():
    A_exact, A_noisy, B_noisy, all_data, model, linear_fit, perfect_fit = build_data()
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=False, return_all=True)

    pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
               label="RANSAC data")

    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, ransac_fit)[:, 0],
               label='RANSAC fit')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, perfect_fit)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, linear_fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()


if __name__ == '__main__':
    draw_data()

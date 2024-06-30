import numpy as np
import scipy as sp
import scipy.linalg as sl
import pylab


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    # 循环次数
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        # 计算最小误差平方和
        test_err = model.get_error(test_points, maybemodel)
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, resids, rank, s = sl.lstsq(A, B)
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model)
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 600  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 30 * np.random.random((n_samples, n_inputs))  # 随机生成0-30之间的600个数据:行向量
    print(A_exact)
    perfect_fit = 70 * np.random.normal(size=(n_inputs, n_outputs))  # 即随机生成一个斜率
    print(perfect_fit)
    B_exact = sp.dot(A_exact, perfect_fit)  # y = k * x
    # 加入高斯噪声,最小二乘法能很好的处理
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 600行一列的数,代表Xi
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # 600行一列的数,代表Yi
    print(A_noisy)
    print(B_noisy)
    # 只有上面都执行成功在会执行if里面的
    if True:
        # 添加局外点
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs)  # 将顺序打乱
        outlier_idxs = all_idxs[:n_outliers]  # 获取前100个点
        A_noisy[outlier_idxs] = 20 * np.random.random(size=(n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))
    # 将A_noisy和B_noisy左右拼接起来
    all_data = np.hstack((A_noisy, B_noisy))
    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False
    # 最小二乘法模型
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    """
    求解线性最小二乘问题 
    最小二乘解：linear_fit
    残差平方和：resids
    系数矩阵秩：rank
    系数矩阵奇异值：s
    """
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # RANSAC算法
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)
    if 1:
        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]
        if 1:
            # 画图
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label='exact system')
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label='linear fit')
        pylab.legend()
        pylab.show()
if __name__ == "__main__":
    test()

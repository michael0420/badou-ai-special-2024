import numpy as np
import scipy.linalg as sl
import pylab

'''
实现ransac
'''

def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    '''
    输入：
        data - 样本点
        model - 确定的模型
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值：作为判断点满足模型的条件
        d - 拟合较好时，需要的样本点最少的个数，当做阈值看待
    输出：
        bestfit - 最优拟合解（返回nil，如果未找到）
    '''
    iterations = 0  # 迭代次数
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_point = data[test_idxs]
        maybe_model = model.fit(maybe_inliers)
        test_err = model.get_error(test_point, maybe_model)  # 计算误差：最小平方和
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  #拼接数组
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  #更新为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  #更新局内点
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


class LinearLeastSquareModel:
    # 最小二乘法求线性解，用于ransac输入模型
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
        B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 残差平方和
        return err_per_point


def test():
    # 生成数据
    n_samples = 500
    n_inputs = 1
    n_outputs = 1
    A_exact = np.random.random((n_samples, n_inputs)) * 20  # 生成0-20之间的500个随机数
    perfect_fit = np.random.normal(size=(n_inputs, n_outputs)) * 60  # 生成随机的斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = k * x
    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    # 添加局外点
    n_outliers = 100
    all_idxs = np.arange(A_noisy.shape[0])
    np.random.shuffle(all_idxs)  # 打乱all_idxs
    outlier_idxs = all_idxs[:n_outliers]  # 100个随机局外点
    A_noisy[outlier_idxs] = np.random.random((n_outliers, n_inputs)) * 20
    B_noisy[outlier_idxs] = np.random.normal(size=(n_outliers, n_outputs)) * 60

    all_data = np.hstack((A_noisy, B_noisy))

    input_columns = range(n_inputs)
    output_columns = [n_inputs + i for i in range(n_outputs)]
    debug = False

    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)

    # 最小二乘法求线性解
    linear_fit, resids, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # ransac
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)


    sort_idxs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_idxs]

    if 1:
        pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
        pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx', label="RANSAC data")
    else:
        pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
        pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

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
    test()

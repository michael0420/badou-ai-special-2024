import numpy as np
import scipy as sp
import scipy.linalg as sl


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    bestfit = None
    best_err = np.inf
    best_idxs = None
    iteration = 0
    while iteration < k:
        # 1选取n个点作为内群数据
        data_size_l = np.arange(data.shape[0])
        np.random.shuffle(data_size_l)
        maybe_idxs = data_size_l[:n]
        test_idxs = data_size_l[n:]
        maybe_inliers = data[maybe_idxs]
        test_liers = data[test_idxs]
        # 2根据n个内群点训练模型
        maybe_model = model.fit(maybe_inliers)
        # 3将外群数据带入模型
        test_errs = model.get_err(test_liers, maybe_model)
        also_idxs = test_idxs[test_errs < t]
        also_liers = data[also_idxs]
        if len(also_liers) > d:
            better_liers = np.concatenate((maybe_inliers, also_liers))
            better_model = model.fit(better_liers)
            better_errs = model.get_err(better_liers, better_model)
            this_err = np.mean(better_errs, axis=0)
            if this_err < best_err:
                best_err = this_err
                bestfit = better_model
                best_idxs = np.concatenate((maybe_idxs, also_idxs))
        iteration += 1
    if bestfit is None:
        raise ValueError('none is fit')
    if return_all is True:
        return bestfit, {'inliner_idxs': best_idxs}
    else:
        return bestfit


class LinearLeastSquareModel:
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        x, r, rank, s = sl.lstsq(A, B)
        return x

    def get_err(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = np.dot(A, model)
        err = np.sum((B_fit - B) ** 2, axis=1)
        return err


def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成[0,20)之间的500个数据:行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    B_exact = np.dot(A_exact, perfect_fit)  # y = x * k，B_exact是y，内群数据

    # 加入高斯噪声,最小二乘能很好的处理，是离群数据
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi，加噪声
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)  # B_exact.shape是500 * 1行向量,代表Yi

    if 1:
        # 添加"局外点"
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 获取索引0-499
        np.random.shuffle(all_idxs)  # 将all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 前100个点。100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入噪声和局外点的Xi，加入噪声和局外点
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入噪声和局外点的Yi，加局外点
    # setup model
    all_data = np.hstack((A_noisy, B_noisy))  # 形式([Xi,Yi]....) shape:(500,2)500行2列，加入噪声和局外点的数据
    input_columns = range(n_inputs)  # 数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 类的实例化:用最小二乘生成已知模型
    # linear_fit是最小二乘法结果
    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

    # run RANSAC 算法
    # def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliner_idxs'], 0], B_noisy[ransac_data['inliner_idxs'], 0], 'bx',
                       label="RANSAC data")
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


if __name__ == "__main__":
    test()

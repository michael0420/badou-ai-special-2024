# 随机采样一致性算法
import numpy as np


def generate_data():
    """
    1、用公司y=kx生成数据基本数据
    2、添加高斯噪声
    3、添加据外点
    """
    # 生成基本数据
    n_sample = 500
    n_input = 1
    n_output = 1
    a_extra = 20 * np.random.random((n_sample, n_input))
    perfect_fit = 60 * np.random.normal(size=(n_input, n_output))
    b_extra = np.dot(a_extra, perfect_fit)

    # 添加高斯噪声
    a_noise = a_extra + np.random.normal(size=a_extra.shape)
    b_noise = b_extra + np.random.normal(size=b_extra.shape)

    # 从原始数据中随机抽取n_outlines个数据替换成据外点
    n_outlines = 100
    all_idxs = np.arange(a_noise.shape[0])
    np.random.shuffle(all_idxs)
    out_liners = all_idxs[:n_outlines]
    a_noise[out_liners] = 20 * np.random.random((n_outlines, n_input))
    b_noise[out_liners] = 20 * np.random.normal(size=(n_outlines, n_output))

    return a_noise, b_noise


def run_ransac():
    a_data, b_data = generate_data()
    n_input = a_data.shape[1]
    n_output = b_data.shape[1]
    all_data = np.hstack((a_data, b_data))
    input_col = range(n_input)
    output_col = [n_input + i for i in range(n_output)]
    debug = False
    model = LinearLeastSquareModel(input_col, output_col, debug=debug)

    linear_fit, res_ids, rank, s = np.linalg.lstsq(all_data[:, input_col], all_data[:, output_col])

    ransac_fit = ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=False)

    print(f'linear_fit：%d', linear_fit)
    print(f'ransac_fit：%d', ransac_fit)


class LinearLeastSquareModel:
    def __init__(self, input_col, output_col, debug=False):
        self.input_col = input_col
        self.output_col = output_col
        self.debug = debug

    def fit(self, data):
        x_data = np.vstack([data[:, i] for i in self.input_col]).T
        y_data = np.vstack([data[:, i] for i in self.output_col]).T
        x, resids, rank, s = np.linalg.lstsq(x_data, y_data)
        return x

    def get_error(self, data, model):
        x_data = np.vstack([data[:, i] for i in self.input_col]).T
        y_data = np.vstack([data[:, i] for i in self.output_col]).T
        fit = np.dot(x_data, model)
        error_per_point = np.sum((y_data - fit) ** 2, axis=1)
        return error_per_point


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def ransac(data, model, n, k, t, d, debug=False, return_all=True):
    iterations = 0
    bestfit = None
    besterr = np.inf  # 设置默认值
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]  # 获取size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 若干行(Xi,Yi)数据点
        maybemodel = model.fit(maybe_inliers)  # 拟合模型
        test_err = model.get_error(test_points, maybemodel)  # 计算误差:平方和最小
        also_idxs = test_idxs[test_err < t]
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


if __name__ == '__main__':
    run_ransac()

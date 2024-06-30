import numpy as np
import scipy as sp
import scipy.linalg as sl

def random_parition(n, n_data):
    allIndexs = np.arange(n_data)
    np.random.shuffle(allIndexs)
    index1 = allIndexs[:n]
    index2 = allIndexs[n:]
    return index1, index2

class LinearLeastSquareModel:
    def __init__(self, inputColumn, outputColumn, debug = False):
        self.inputColumn = inputColumn
        self.outputColumn = outputColumn
        self.debug = debug

    def fit(self, data):
        A = np.vstack([data[:,i] for i in self.inputColumn]).T
        B = np.vstack([data[:,i] for i in self.outputColumn]).T
        x, resides, rank, s = sl.lstsq(A, B)#回归系数、残差平方和、自变量X的秩、X的奇异值。
        return x

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.inputColumn]).T
        B = np.vstack([data[:, i] for i in self.outputColumn]).T
        B_fit = np.dot(A, model)#计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

def ransac(data, model, n, k, t, d, debug = False, return_all = False):
    '''
    :param data: 样本点
    :param model: 模型
    :param n:     样本点
    :param k:     最大迭代次数
    :param t:     阈值
    :param d:     拟合较好时，需要的样本点最小的个数，当作阈值看待
    :param debug:
    :param return_all:
    :return: bestfit : 最优拟合解
    '''

    iterations = 0
    bestfit = None
    besterr = np.inf #极大值
    best_in_indexs = None

    while iterations < k :
        maybe_indexs, test_indexs = random_parition(n, data.shape[0])
        maybe_inliers = data[maybe_indexs, :]
        test_points = data[test_indexs]
        maybeModel = model.fit(maybe_inliers)#拟合
        test_err = model.get_error(test_points, maybeModel)#计算平方和
        also_indexs = test_indexs[test_err < t]
        also_inliers = data[also_indexs, :]

        if(len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))
            bettermodel = model.fit(betterdata)
            betterErrs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(betterErrs)#平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_in_indexs = np.concatenate((maybe_indexs, also_indexs))
        iterations += 1

    if bestfit is None:
        raise  ValueError("its error")
    if return_all:
        return  bestfit,{'inliers':best_in_indexs}
    else:
        return  bestfit

def test():
    n_samples = 500
    n_input = 1
    n_output = 1
    A_exact = 20 * np.random.random((n_samples, n_input))
    fit = 60 * np.random.normal(size=(n_input, n_output))
    B_exact = np.dot(A_exact, fit)

    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    #添加外点
    n_outliers = 100
    allIndex = np.arange(A_noisy.shape[0])
    np.random.shuffle(allIndex)
    outLier_indexs = allIndex[:n_outliers]
    A_noisy[outLier_indexs] = 20 * np.random.random((n_outliers, n_input))
    B_noisy [outLier_indexs] = 50 * np.random.random(size = (n_outliers, n_output))

    all_data = np.hstack((A_noisy, B_noisy))
    input_cols = range(n_input)
    output_cols = [n_input + i for i in range(n_output)]
    debug = False
    model = LinearLeastSquareModel(input_cols, output_cols, debug=debug)

    linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:,input_cols], all_data[:,output_cols])

    ransac_fit, ransac_data = ransac(all_data, model, 50, 1000, 7e3, 300, debug = debug, return_all= True)

    sort_indexs = np.argsort(A_exact[:, 0])
    A_col0_sorted = A_exact[sort_indexs]
    import pylab

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
               np.dot(A_col0_sorted, fit)[:, 0],
               label='exact system')
    pylab.plot(A_col0_sorted[:, 0],
               np.dot(A_col0_sorted, fit)[:, 0],
               label='linear fit')
    pylab.legend()
    pylab.show()

if __name__ == "__main__":
    test()